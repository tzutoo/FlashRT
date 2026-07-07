#!/usr/bin/env python3
"""Phase A A/B bench: --max-seq 196608 (192K, current) vs 229376 (224K, new).

WSL2-aware variant: uses 'docker exec <name> curl ...' to make HTTP calls,
because WSL2 host networking doesn't expose the container's host-network
port 8000 to localhost on the WSL side. The container is started with
--network=host so the server's port 8000 is bound inside the container
network namespace; we run curl from inside that namespace.

Goal: confirm the move 192K -> 224K has zero decode regression on the
realistic agent context sizes (1K, 8K, 16K, 32K). VRAM headroom is
also reported so we can see how close to the cliff the new config sits.
"""
from __future__ import annotations
import json, subprocess, time, statistics

NVFP4 = "/home/tzuto/Projects/FlashRT/qwen36_nvfp4"
FP8 = "/home/tzuto/Projects/FlashRT/qwen36_fp8"

# (label, image, max_seq)
CONFIGS = [
    ("192K_current", "flashrt-server:5090",       196608),
    ("224K_new",     "flashrt-server:5090-224k",  229376),
]
CTXS = [1024, 8192, 16384, 32768]
SAMPLES = 3
OUT = 100   # generation tokens per request

def sh(cmd, check=False):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and r.returncode != 0:
        print(f"  CMD FAILED: {cmd}\n  stderr: {r.stderr[:300]}")
    return r

def start_container(image: str, max_seq: int, name: str) -> bool:
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
    # copy the runner script into the container (avoids shell escaping)
    sh(f"docker cp /tmp/phaseA_runner.py {name}:/tmp/phaseA_runner.py")
    # wait for /health (called from inside the container's network namespace)
    for _ in range(180):
        r = sh(
            f"docker exec {name} curl -s -m 3 http://127.0.0.1:8000/health"
        )
        if r.returncode == 0 and r.stdout.strip():
            try:
                d = json.loads(r.stdout)
                if d.get("status") == "ok":
                    return True
            except Exception:
                pass
        time.sleep(1)
    print("  health check timed out (180s)")
    sh(f"docker logs --tail 30 {name}")
    return False

def free_mib() -> int:
    o = sh("nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits").stdout.strip()
    return int(o) if o.isdigit() else -1

def measure(name: str, ctx: int) -> float:
    """One sample: cold prefill of `ctx` tokens, then `OUT` decode tokens.
    Called from inside the container's network namespace via docker exec
    (we docker-cp the request body + a runner script to avoid shell escaping)."""
    reps = ctx * 4 // 66 + 1
    f = "The quick brown fox jumps over the lazy dog near the old mill. " * reps
    body = {
        "model": "qwen36-27b",
        "messages": [{"role": "user", "content": f + "\nCount from 1 to 100, one per line."}],
        "max_tokens": OUT,
        "stream": False,
        "flashrt_session_id": f"pA-{ctx}-{time.time_ns()}",
        "flashrt_cache_salt":  f"pA-{ctx}-{time.time_ns()}",
    }
    host_tmp = f"/tmp/phaseA_req_{ctx}_{int(time.time()*1000)}.json"
    with open(host_tmp, "w") as fp:
        json.dump(body, fp)
    sh(f"docker cp {host_tmp} {name}:/tmp/req.json")
    subprocess.run(f"rm -f {host_tmp}", shell=True)
    # use a pre-written script in the container (cp'd at startup) to avoid escaping
    r = sh(f"docker exec {name} python3 /tmp/phaseA_runner.py")
    if r.returncode != 0:
        raise RuntimeError(f"request failed: {r.stderr[:200]}")
    d = json.loads(r.stdout)
    return d["flashrt"]["decode_tok_per_s"]

def delete_all_sessions(name: str):
    try:
        r = sh(f"docker exec {name} curl -s -m 3 http://127.0.0.1:8000/health")
        h = json.loads(r.stdout)
        for s in h.get("sessions", {}).get("sessions", []):
            sh(f"docker exec {name} curl -s -X DELETE "
               f"http://127.0.0.1:8000/v1/sessions/{s.get('session_id')} >/dev/null")
    except Exception:
        pass

def bench_one_config(label: str, image: str, max_seq: int) -> dict:
    name = f"flashrt-phaseA-{label}"
    print(f"\n[{label}] starting container (image={image}, max_seq={max_seq}) ...")
    if not start_container(image, max_seq, name):
        return {"label": label, "max_seq": max_seq, "error": "start failed"}
    time.sleep(3)
    free = free_mib()
    print(f"[{label}] free VRAM: {free} MiB")
    delete_all_sessions(name)
    results = {}
    for ctx in CTXS:
        samples = []
        for i in range(SAMPLES):
            try:
                s = measure(name, ctx)
                samples.append(s)
                print(f"  ctx={ctx:>5}  sample {i+1}/{SAMPLES}: {s:.2f} tok/s")
            except Exception as e:
                print(f"  ctx={ctx:>5}  sample {i+1}/{SAMPLES}: FAILED ({e})")
                samples.append(-1)
            delete_all_sessions(name)
        valid = [s for s in samples if s > 0]
        med = statistics.median(valid) if valid else -1
        results[ctx] = med
    sh(f"docker rm -f {name} >/dev/null 2>&1")
    time.sleep(3)
    return {"label": label, "max_seq": max_seq, "free_mib": free, "results": results}

def main():
    print(f"Phase A: --max-seq 192K vs 224K A/B (Qwen3.6-27B NVFP4, RTX 5090)")
    print(f"  contexts: {CTXS}   samples per (config, ctx): {SAMPLES} (median)")
    print(f"  out tokens per request: {OUT}")
    print("="*78)
    all_results = []
    for label, image, max_seq in CONFIGS:
        all_results.append(bench_one_config(label, image, max_seq))
    print("\n" + "="*78)
    print("RESULTS - decode tok/s (median of 3 samples)")
    print("="*78)
    print(f"{'config':>14}  {'max_seq':>8}  {'free MiB':>9}  "
          + "  ".join(f"{c//1024:>4}K" for c in CTXS))
    print("-"*78)
    for r in all_results:
        if "error" in r:
            print(f"{r['label']:>14}  {r['max_seq']:>8}  {'ERR':>9}")
            continue
        row = f"{r['label']:>14}  {r['max_seq']:>8}  {r['free_mib']:>9}  "
        row += "  ".join(f"{r['results'].get(c, -1):>5.1f}" for c in CTXS)
        print(row)
    if len(all_results) == 2 and "error" not in all_results[1]:
        a, b = all_results[0]["results"], all_results[1]["results"]
        print("-"*78)
        delta_row = f"{'delta (224-192)':>14}  {'':>8}  {'':>9}  "
        delta_row += "  ".join(
            f"{(b[c]-a[c]):+5.1f}" if a.get(c, -1) > 0 and b.get(c, -1) > 0 else "  n/a"
            for c in CTXS
        )
        print(delta_row)
        pct_row = f"{'pct change':>14}  {'':>8}  {'':>9}  "
        pct_row += "  ".join(
            f"{(b[c]-a[c])/a[c]*100:+5.1f}%" if a.get(c, -1) > 0 and b.get(c, -1) > 0 else "  n/a"
            for c in CTXS
        )
        print(pct_row)
    out = f"benchmarks/phaseA_ab_results_{int(time.time())}.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nraw results saved to {out}")

if __name__ == "__main__":
    main()
