#!/usr/bin/env python3
"""Simulate pi's long growing conversation AT a given max_seq.

Grows a single session from small to ~50K context (like a real long coding
session), measuring decode tok/s + free VRAM at each turn. If decode stays fast
all the way up, then this max_seq does NOT need restarts under pi load. If it
degrades as context grows, restarts are still needed.

Usage: python3 verify_pi_growth.py <max_seq_label>
"""
from __future__ import annotations
import json, subprocess, time, urllib.request

BASE = "http://127.0.0.1:8000"

def free_mib():
    o = subprocess.check_output(
        ["nvidia-smi","--query-gpu=memory.free","--format=csv,noheader,nounits"]).decode().strip()
    return int(o) if o.isdigit() else -1

def post(body, timeout=300):
    req = urllib.request.Request(f"{BASE}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type":"application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def grow_session(label, max_turns=8):
    sid = f"growth-{label}-{time.time_ns()}"
    # Start with a big-ish user message so the session grows quickly toward
    # ~50K (mirrors a long coding session with file pastes).
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content":
            "Here is a large codebase context:\n\n" +
            ("def example_function(x, y):\n    return x + y\n" * 800) +
            "\n\nExplain the structure."},
    ]
    print(f"=== {label} (session grows each turn) ===", flush=True)
    for t in range(1, max_turns + 1):
        body = {"model":"qwen36-27b","messages":[dict(m) for m in messages],
                "max_tokens":80,"stream":False,
                "flashrt_session_id":sid,"flashrt_cache_salt":""}
        res = post(body)
        f = res.get("flashrt", {})
        ptok = res["usage"]["prompt_tokens"]
        text = (res["choices"][0]["message"]["content"] or "")[:40].replace("\n"," ")
        print(f"  turn {t}: p={ptok:>6} new_prefill={f.get('new_prefill_tokens',0):>4} "
              f"decode={f.get('decode_tok_per_s',0):5.1f} tok/s  free={free_mib():>5} MiB  ({text}...)",
              flush=True)
        # append assistant + a new chunky user turn (grows the session)
        messages.append({"role":"assistant","content":res["choices"][0]["message"]["content"]})
        messages.append({"role":"user","content":
            "Now consider this additional module:\n\n" +
            ("class Helper:\n    pass\n" * 400) +
            "\nHow does it interact?"})

if __name__ == "__main__":
    import sys
    grow_session(sys.argv[1] if len(sys.argv) > 1 else "test", max_turns=14)
