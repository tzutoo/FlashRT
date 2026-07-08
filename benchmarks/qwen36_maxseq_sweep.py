#!/usr/bin/env python3
"""Sweep --max-seq with MULTIPLE samples per context to find the real sweet spot.

Takes N samples at each (max_seq, ctx) and reports the median, so a single
lucky/unlucky sample can't mislead (the failure mode that made 245760 look
fast in an earlier single-sample sweep). Finds the largest max_seq that keeps
fast decode at realistic agent contexts (1K, 8K, 16K, 32K).
"""
from __future__ import annotations
import json, subprocess, time, urllib.request, statistics

NVFP4 = "/home/tzuto/Projects/FlashRT/qwen36_nvfp4"
FP8 = "/home/tzuto/Projects/FlashRT/qwen36_fp8"
IMAGE = "flashrt-server:5090"
MAXSEQS = [32768, 65536, 98304, 131072, 163840, 196608, 229376, 245760]
CTXS = [1024, 8192, 16384, 32768]
SAMPLES = 3
OUT = 100

def sh(cmd): return subprocess.run(cmd, shell=True, capture_output=True, text=True)

def restart(maxseq):
    sh("docker rm -f flashrt-qwen36 2>/dev/null")
    r = sh(f"docker run --gpus all --ipc=host "
           f"--ulimit memlock=-1 --ulimit stack=67108864 "
           f"--stop-timeout 30 -p 8765:8000 -d --name flashrt-qwen36 "
           f"-v {NVFP4}:/nvfp4:ro -v {FP8}:/fp8:ro "
           f"-e FLASHRT_QWEN36_MTP_CKPT_DIR=/fp8 -e FLASHRT_QWEN36_LONG_KV_CACHE=fp8 "
           f"-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
           f"-e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 "
           f"{IMAGE} python3 -m serving.qwen36_agent.server --checkpoint /nvfp4 "
           f"--max-seq {maxseq} --route-min-seq 0 "
           f"--default-max-tokens 8192 --max-output-tokens 65536 "
           f"--host 0.0.0.0 --port 8000")
    if r.returncode != 0: return False
    for _ in range(120):
        try:
            req = urllib.request.Request("http://127.0.0.1:8765/v1/chat/completions",
                data=json.dumps({"model":"qwen36-27b","messages":[{"role":"user","content":"hi"}],"max_tokens":3}).encode(),
                headers={"Content-Type":"application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=60) as resp: resp.read()
            return True
        except Exception: time.sleep(1)
    return False

def free_mib():
    o = sh("nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits").stdout.strip()
    return int(o) if o.isdigit() else -1

def measure(ctx):
    reps = ctx * 4 // 66 + 1
    f = "The quick brown fox jumps over the lazy dog near the old mill. " * reps
    body = {"model":"qwen36-27b","messages":[{"role":"user","content":f+"\nCount from 1 to 100, one per line."}],
            "max_tokens":OUT,"stream":False,
            "flashrt_session_id":f"m{ctx}-{time.time_ns()}","flashrt_cache_salt":f"m{ctx}-{time.time_ns()}"}
    req = urllib.request.Request("http://127.0.0.1:8765/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type":"application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=400) as r: d = json.loads(r.read())
    return d["flashrt"]["decode_tok_per_s"]

def delete_all():
    try:
        with urllib.request.urlopen("http://127.0.0.1:8765/health",timeout=5) as r: h=json.loads(r.read())
        for s in h.get("sessions",{}).get("sessions",[]):
            sh(f"curl -s -X DELETE http://127.0.0.1:8765/v1/sessions/{s.get('session_id')} >/dev/null")
    except Exception: pass

def main():
    print(f"{'max_seq':>8} {'free_MiB':>9}  " + "  ".join(f"{c//1024:>4}K_med" for c in CTXS))
    print("-"*60)
    for ms in MAXSEQS:
        if not restart(ms):
            print(f"{ms:>8}  START FAILED"); continue
        free = free_mib()
        delete_all()
        meds = []
        for ctx in CTXS:
            samples = []
            for _ in range(SAMPLES):
                try: samples.append(measure(ctx))
                except Exception: samples.append(-1)
                delete_all()
            meds.append(statistics.median(samples) if samples else -1)
        print(f"{ms:>8} {free:>9}  " + "  ".join(f"{m:>8.1f}" for m in meds))
        sh("docker rm -f flashrt-qwen36 >/dev/null 2>&1")
        time.sleep(3)
    restart(32768)
    print("\nrestored: --max-seq 32768")

if __name__ == "__main__":
    main()
