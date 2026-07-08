#!/usr/bin/env python3
"""Simulate pi/agent multi-turn traffic: a growing conversation.

Each turn appends the assistant's previous reply + a new user message, so the
prompt grows monotonically (like a real agent session). Uses ONE session so KV
cache appends (realistic). Reports per-turn decode tok/s from the server stats.

This is the honest test for the graphs-ON-vs-OFF decision: a benchmark reuses
the *same* prompt (positions repeat -> warm replay), but an agent's decode
positions are always new. So if graphs ON gives ~32 tok/s here (capture) vs
graphs OFF ~130 tok/s, graphs must stay OFF for serving regardless of max_seq.

Run once per config (restart the container between).
"""
from __future__ import annotations
import json, time, urllib.request

BASE = "http://127.0.0.1:8765"

def post(body, timeout=300):
    req = urllib.request.Request(f"{BASE}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def run(label, turns=5):
    sid = f"agent-sim-{label}-{time.time_ns()}"
    messages = [{"role": "system", "content": "You are a helpful coding assistant."},
                {"role": "user", "content": "Explain how quicksort works, with a small example."}]
    print(f"=== {label} (session {sid[:18]}) ===", flush=True)
    for t in range(1, turns + 1):
        body = {"model": "qwen36-27b", "messages": [dict(m) for m in messages],
                "max_tokens": 150, "stream": False,
                "flashrt_session_id": sid, "flashrt_cache_salt": ""}
        t0 = time.perf_counter()
        res = post(body)
        f = res.get("flashrt", {})
        text = res["choices"][0]["message"]["content"] or ""
        print(f"  turn {t}: p={res['usage']['prompt_tokens']:>5} out={res['usage']['completion_tokens']:>3} "
              f"new_prefill={f.get('new_prefill_tokens',0):>3} "
              f"decode={f.get('decode_tok_per_s',0):5.1f} tok/s", flush=True)
        # grow the conversation: append assistant reply + a new user turn
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user",
                         "content": f"Now show me turn {t+1}: contrast it with mergesort in one paragraph."})

if __name__ == "__main__":
    import sys
    run(sys.argv[1] if len(sys.argv) > 1 else "config")
