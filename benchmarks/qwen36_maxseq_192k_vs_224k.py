#!/usr/bin/env python3
"""qwen36_maxseq_192k_vs_224k.py — definitive max_seq sweet-spot bench.

Tests 192K (recommended) vs 224K (alternative) at both short (1K-32K) and
long (64K-192K) input contexts. Same methodology as qwen36_phaseA_ab.py
plus long-context coverage that the Phase A bench was missing.

The short-context sweep runs 3 samples per (config, ctx) for noise
reduction. The long-context sweep runs 1 sample per (config, ctx)
because each request is 8-60 seconds.

Output: writes maxseq_192k_vs_224k_results_<ts>.json with full data.

Usage:
  python3 benchmarks/qwen36_maxseq_192k_vs_224k.py

Findings (RTX 5090, Qwen3.6-27B NVFP4, FP8-KV long route, 2026-07-08):
  - 192K and 224K are equivalent at all contexts (within 0-6%)
  - 256K is in the cliff (decode drops to 7-15 tok/s) — excluded by default
  - Decode degrades with input context (O(S²) attention) regardless of
    max_seq: 1K=135, 32K=127, 64K=82, 128K=66, 192K=40 tok/s
  - 192K is the recommended sweet spot (more cliff headroom, 6% faster
    at 192K input context)
"""
from __future__ import annotations
import json, subprocess, time, statistics, sys
sys.stdout.reconfigure(line_buffering=True)

NVFP4 = "/home/tzuto/Projects/FlashRT/qwen36_nvfp4"
FP8   = "/home/tzuto/Projects/FlashRT/qwen36_fp8"

# (label, max_seq)
CONFIGS = [
    ("192K", 196608),   # recommended — reverts to original
    ("224K", 229376),   # Phase A "free upgrade" — equivalent to 192K
]

# Short contexts: 3 samples each (median), for noise reduction
SHORT_CTXS = [1024, 8192, 16384, 32768]
SHORT_SAMPLES = 3
SHORT_OUT = 100

# Long contexts: 1 sample each (these are SLOW: 8-60s per request)
LONG_CTXS = [65536, 131072, 196608]
LONG_SAMPLES = 1
LONG_OUT = 100

REQUEST_TIMEOUT = 240  # 4 min per long-context request

def sh(cmd, check=False):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
    if check and r.returncode != 0:
        print(f"  CMD FAILED: {cmd}\n  stderr: {r.stderr[:300]}", flush=True)
    return r

def start_container(image, max_seq, name) -> bool:
    sh(f"docker rm -f {name} 2>/dev/null")
    sh(
        f"docker run --restart always --gpus all --network=host --ipc=host "
        f"--ulimit memlock=-1 --ulimit stack=67108864 "
        f"--stop-timeout 30 -d --name {name} "
        f"-v {NVFP4}:/nvfp4:ro -v {FP8}:/fp8:ro "
        f"-v /tmp:/host_tmp "
        f"-e FLASHRT_QWEN36_MTP_CKPT_DIR=/fp8 "
        f"-e FLASHRT_QWEN36_LONG_KV_CACHE=fp8 "
        f"-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
        f"-e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 "
        f"{image} python3 -m serving.qwen36_agent.server "
        f"--checkpoint /nvfp4 --max-seq {max_seq} --route-min-seq 0 "
        f"--default-max-tokens 32768 --max-output-tokens 65536 "
        f"--host 0.0.0.0 --port 8000",
        check=True,
    )
    sh(f"docker cp /tmp/phaseA_runner.py {name}:/tmp/phaseA_runner.py")
    for _ in range(150):
        try:
            r = sh(f"docker exec {name} curl -s -m 3 http://127.0.0.1:8000/health")
            if r.returncode == 0 and r.stdout.strip():
                if json.loads(r.stdout).get("status") == "ok":
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False

def free_mib() -> int:
    o = sh("nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits").stdout.strip()
    return int(o) if o.isdigit() else -1

def measure(name, ctx, out_tokens) -> dict:
    reps = ctx * 4 // 66 + 1
    f = "The quick brown fox jumps over the lazy dog near the old mill. " * reps
    body = {
        "model": "qwen36-27b",
        "messages": [{"role": "user",
                      "content": f + "\nCount from 1 to 100, one per line."}],
        "max_tokens": out_tokens,
        "stream": False,
        "flashrt_session_id": f"maxseq-{ctx}-{time.time_ns()}",
        "flashrt_cache_salt":  f"maxseq-{ctx}-{time.time_ns()}",
    }
    host_tmp = f"/tmp/maxseq_req_{ctx}_{int(time.time()*1000)}.json"
    with open(host_tmp, "w") as fp:
        json.dump(body, fp)
    sh(f"docker cp {host_tmp} {name}:/tmp/req.json")
    subprocess.run(f"rm -f {host_tmp}", shell=True)
    t0 = time.perf_counter()
    r = sh(f"timeout {REQUEST_TIMEOUT} docker exec {name} python3 /tmp/phaseA_runner.py")
    wall_s = time.perf_counter() - t0
    if r.returncode != 0:
        raise RuntimeError(f"request failed (rc={r.returncode}, wall={wall_s:.1f}s)")
    d = json.loads(r.stdout)
    return {
        "wall_s": wall_s,
        "decode_tok_per_s": d["flashrt"]["decode_tok_per_s"],
        "prefill_ms": d["flashrt"]["prefill_ms"],
        "decode_ms": d["flashrt"]["decode_ms"],
        "out_tokens": d["usage"]["completion_tokens"],
    }

def delete_all_sessions(name):
    try:
        r = sh(f"docker exec {name} curl -s -m 3 http://127.0.0.1:8000/health")
        h = json.loads(r.stdout)
        for s in h.get("sessions", {}).get("sessions", []):
            sh(f"docker exec {name} curl -s -X DELETE "
               f"http://127.0.0.1:8000/v1/sessions/{s.get('session_id')} >/dev/null")
    except Exception:
        pass

def bench_config(label, max_seq, ctxs, samples, out_tokens) -> dict:
    name = f"flashrt-maxseq-{label}"
    print(f"\n[{label}] starting (max_seq={max_seq}) ...", flush=True)
    if not start_container("flashrt-server:5090", max_seq, name):
        return {"label": label, "max_seq": max_seq, "error": "start failed"}
    time.sleep(3)
    free = free_mib()
    print(f"[{label}] free VRAM: {free} MiB", flush=True)
    delete_all_sessions(name)
    results = {}
    for ctx in ctxs:
        per_ctx_samples = []
        for i in range(samples):
            try:
                t0 = time.perf_counter()
                m = measure(name, ctx, out_tokens)
                wall = time.perf_counter() - t0
                per_ctx_samples.append(m["decode_tok_per_s"])
                print(f"  ctx={ctx//1024:>3}K  sample {i+1}/{samples}: "
                      f"decode={m['decode_tok_per_s']:6.2f} tok/s  "
                      f"prefill={m['prefill_ms']/1000:6.1f}s  "
                      f"wall={wall:6.1f}s", flush=True)
            except Exception as e:
                print(f"  ctx={ctx//1024:>3}K  sample {i+1}/{samples}: FAILED ({e})",
                      flush=True)
            delete_all_sessions(name)
            time.sleep(2)
        if per_ctx_samples:
            results[ctx] = {
                "median_tok_per_s": statistics.median(per_ctx_samples),
                "samples": per_ctx_samples,
            }
    sh(f"docker rm -f {name} >/dev/null 2>&1")
    time.sleep(3)
    return {"label": label, "max_seq": max_seq, "free_mib": free, "results": results}

def main():
    print("qwen36 max_seq sweet-spot bench: 192K vs 224K at short AND long contexts",
          flush=True)
    print(f"  short ctxs: {SHORT_CTXS}  ({SHORT_SAMPLES} samples each)",
          flush=True)
    print(f"  long  ctxs: {LONG_CTXS}   ({LONG_SAMPLES} sample each — slow)",
          flush=True)
    print("="*80, flush=True)
    all_results = []
    for label, max_seq in CONFIGS:
        print(f"\n--- {label} short contexts ---", flush=True)
        short = bench_config(label, max_seq, SHORT_CTXS, SHORT_SAMPLES, SHORT_OUT)
        if "error" not in short:
            print(f"\n--- {label} long contexts ---", flush=True)
            long = bench_config(label, max_seq, LONG_CTXS, LONG_SAMPLES, LONG_OUT)
            short["long_results"] = long.get("results", {})
        all_results.append(short)

    # Comparison table
    print("\n" + "="*80, flush=True)
    print("RESULTS — decode tok/s (median of 3 for short, 1 for long)", flush=True)
    print("="*80, flush=True)
    all_ctxs = SHORT_CTXS + LONG_CTXS
    print(f"{'config':<8}  {'max_seq':<8}  {'free MiB':<9}  "
          + "  ".join(f"{c//1024:>3}K" for c in all_ctxs), flush=True)
    print("-"*80, flush=True)
    for r in all_results:
        if "error" in r:
            print(f"{r['label']:<8}  {r['max_seq']:<8}  {'ERR':<9}", flush=True)
            continue
        row = f"{r['label']:<8}  {r['max_seq']:<8}  {r.get('free_mib', 'N/A'):<9}  "
        cells = []
        for c in SHORT_CTXS:
            res = r['results'].get(c, {})
            tps = res.get("median_tok_per_s", -1) if isinstance(res, dict) else -1
            cells.append(f"{tps:>4.1f}" if tps > 0 else "  n/a")
        for c in LONG_CTXS:
            res = r.get('long_results', {}).get(c, {})
            tps = res.get("median_tok_per_s", -1) if isinstance(res, dict) else -1
            cells.append(f"{tps:>4.1f}" if tps > 0 else "  n/a")
        row += "  ".join(cells)
        print(row, flush=True)

    # Delta
    if len(all_results) == 2 and "error" not in all_results[1]:
        a_short = all_results[0]["results"]
        a_long  = all_results[0].get("long_results", {})
        b_short = all_results[1]["results"]
        b_long  = all_results[1].get("long_results", {})
        print("-"*80, flush=True)
        delta_row = f"{'delta':<8}  {'':<8}  {'':<9}  "
        cells = []
        for c in SHORT_CTXS + LONG_CTXS:
            src = a_short.get(c) if c in SHORT_CTXS else a_long.get(c)
            tgt = b_short.get(c) if c in SHORT_CTXS else b_long.get(c)
            a_tps = src.get("median_tok_per_s") if src else None
            b_tps = tgt.get("median_tok_per_s") if tgt else None
            if a_tps and b_tps and a_tps > 0:
                cells.append(f"{b_tps - a_tps:+4.1f}")
            else:
                cells.append("  n/a")
        delta_row += "  ".join(cells)
        print(delta_row, flush=True)

    out = f"benchmarks/maxseq_192k_vs_224k_results_{int(time.time())}.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nraw results saved to {out}", flush=True)

if __name__ == "__main__":
    main()
