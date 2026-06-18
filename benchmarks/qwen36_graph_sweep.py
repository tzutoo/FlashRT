#!/usr/bin/env python3
"""Benchmark Qwen3.6 decode speed: CUDA graphs ON vs OFF across context sizes.

What it measures
----------------
For each target context size it sends two *independent* requests that share an
identical prompt, and reports the decode tok/s reported by the server:

  run 1 ("cold")  — first visit to these decode positions.
                    With graphs OFF this is the steady-state decode speed.
                    With graphs ON every position is a graph *capture* here.
  run 2 ("warm")  — second visit to the SAME positions (identical prompt).
                    With graphs OFF ≈ same as cold (no graph involved).
                    With graphs ON these are graph *replays* (fast).

Each request uses a fresh ``flashrt_session_id`` + unique ``flashrt_cache_salt``
so the prefix/KV cache never reuses across requests — both runs cold-prefill.
The CUDA-graph cache is global to the frontend, so run 2 still hits the warm
graphs captured by run 1. This is the cleanest way to see the cold-capture
penalty vs warm-replay benefit independently of session reuse.

Run it once per graph mode. The server's current graph mode is printed from
``/health`` at the start. To switch modes, restart the container with:

  graphs OFF (production agent default, already the default):
      docker run ... flashrt-server:5090
      (FLASHRT_QWEN36_TQ_VERIFY_GRAPH defaults to 0)

  graphs ON (benchmark / fixed-shape demo):
      docker run ... -e FLASHRT_QWEN36_TQ_VERIFY_GRAPH=1 \
          -e FLASHRT_QWEN36_TQ_MTP_CHAIN_GRAPH=1 \
          flashrt-server:5090 --max-seq 32768 --route-min-seq 0

Usage:
    python3 benchmarks/qwen36_graph_sweep.py --url http://127.0.0.1:8000
        --contexts 1024 4096 16384 32768 --out-tokens 128
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

# A neutral filler whose token length scales predictably with repetition.
# ~44 chars/rep, ~10-11 tokens/rep for Qwen3.6's BPE on English.
_FILLER = "The quick brown fox jumps over the lazy dog near the old mill. "


def _post(url: str, body: dict, timeout: float) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _get(url: str, timeout: float = 5.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read())


def _build_prompt(target_tokens: int) -> str:
    """Rough char→token estimate (≈4 chars/token) then let the server report
    the exact count. The exactness doesn't matter — only that run 1 and run 2
    use the SAME string. The instruction elicits a predictable-length numeric
    list so the model reliably produces close to ``out_tokens`` (a clean
    decode measurement) instead of early-EOS on bare filler."""
    reps = max(1, int(target_tokens * 4 / len(_FILLER)) + 1)
    filler = f"Context target {target_tokens}. " + _FILLER * reps
    return (filler + "\n\nIgnore the above filler. Now count from 1 to 200, "
            "one number per line, nothing else.")


def _delete_session(base: str, session_id: str) -> None:
    """Drop a session's KV cache so a long sweep does not starve VRAM.

    Each benchmark request cold-prefills a fresh session (unique id + salt),
    allocating KV cache for its full prompt length. Without cleanup, a sweep
    across 1K/4K/16K/32K contexts accumulates large KV allocations that leave
    the GPU memory-starved for attention — decode then collapses from ~96
    tok/s to ~17 tok/s (the VRAM-starvation slowdown BUILD_NOTES warns about).
    Deleting the session after we have its stats reclaims that KV memory.
    """
    try:
        req = urllib.request.Request(f"{base}/v1/sessions/{session_id}",
                                     method="DELETE")
        urllib.request.urlopen(req, timeout=10.0).read()
    except Exception:
        pass  # best-effort; not worth aborting a sweep over


def _one_request(base: str, model: str, prompt: str, out_tokens: int,
                 tag: str, delete_after: bool = True) -> dict:
    """Single non-streaming completion. Returns the ``flashrt`` stats dict.
    Unique session_id + cache_salt every call => no prefix reuse => cold
    prefill. The graph cache (global) still persists across calls."""
    session_id = f"bench-{tag}-{int(time.time()*1000)}"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": out_tokens,
        "stream": False,
        "flashrt_session_id": session_id,
        "flashrt_cache_salt": f"salt-{tag}-{time.time_ns()}",
    }
    t0 = time.perf_counter()
    res = _post(f"{base}/v1/chat/completions", body, timeout=600.0)
    wall = time.perf_counter() - t0
    stats = dict(res.get("flashrt") or {})
    stats["wall_ms"] = wall * 1000.0
    stats["prompt_tokens"] = res.get("usage", {}).get("prompt_tokens")
    stats["completion_tokens"] = res.get("usage", {}).get("completion_tokens")
    stats["session_id"] = session_id
    if delete_after:
        _delete_session(base, session_id)
    return stats


def _fmt_speed(v) -> str:
    try:
        return f"{float(v):6.1f}"
    except (TypeError, ValueError):
        return "    --"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000",
                    help="Server base URL (default: %(default)s)")
    ap.add_argument("--model", default="qwen36-27b")
    ap.add_argument("--contexts", type=int, nargs="+",
                    default=[1024, 4096, 16384, 32768],
                    help="Target prompt token sizes")
    ap.add_argument("--out-tokens", type=int, default=128,
                    help="Generated tokens per request (decode window)")
    ap.add_argument("--repeats", type=int, default=1,
                    help="Cold/warm pairs to average per context size")
    ap.add_argument("--no-warm", action="store_true",
                    help="Skip the warm (run 2) pass — measure cold only")
    args = ap.parse_args()

    # Server liveness + graph mode.
    try:
        health = _get(f"{args.url}/health", timeout=10.0)
    except urllib.error.URLError as exc:
        print(f"ERROR: cannot reach {args.url}/health: {exc}", file=sys.stderr)
        return 1
    # Decode graph mode isn't in /health directly; read it from the env that the
    # agent host logs at startup. We just print what we know and let the user
    # confirm which container config is running.
    print(f"server      : {args.url}")
    print(f"model       : {health.get('model')}")
    print(f"max_seq     : {health.get('max_seq')}")
    print(f"speculative : {health.get('speculative')}")
    print(f"out_tokens  : {args.out_tokens}")
    print(f"contexts    : {args.contexts}")
    print(f"note        : confirm graph mode in `docker logs` -> "
          f"'agent decode graph mode: verify_graph=...'")
    print()

    rows = []
    for target in args.contexts:
        prompt = _build_prompt(target)
        for rep in range(args.repeats):
            cold = _one_request(args.url, args.model, prompt,
                                args.out_tokens, f"c{target}-r{rep}-cold")
            warm = None
            if not args.no_warm:
                # Identical prompt => same decode positions => warm graph replay.
                warm = _one_request(args.url, args.model, prompt,
                                    args.out_tokens, f"c{target}-r{rep}-warm")
            rows.append((target, cold, warm, rep))
            ptok = cold.get("prompt_tokens", "?")
            cold_s = _fmt_speed(cold.get("decode_tok_per_s"))
            warm_s = _fmt_speed((warm or {}).get("decode_tok_per_s"))
            print(f"  ctx~{target:<7} prompt_tok={ptok:<7} "
                  f"decode tok/s: cold={cold_s}  warm={warm_s}")
        sys.stdout.flush()

    print()
    header = (f"{'ctx':>7} {'prompt':>7} {'out':>5} "
              f"{'cold tok/s':>11} {'warm tok/s':>11} "
              f"{'cold prefill ms':>16}")
    print(header)
    print("-" * len(header))
    for target, cold, warm, _rep in rows:
        ptok = cold.get("prompt_tokens")
        otok = cold.get("completion_tokens")
        cold_s = _fmt_speed(cold.get("decode_tok_per_s"))
        warm_s = _fmt_speed((warm or {}).get("decode_tok_per_s"))
        pre_ms = cold.get("prefill_ms")
        pre_ms_s = f"{float(pre_ms):8.0f}" if pre_ms is not None else "      --"
        print(f"{target:>7} {str(ptok):>7} {str(otok):>5} "
              f"{cold_s:>11} {warm_s:>11} {pre_ms_s:>16}")

    print()
    print("How to read it:")
    print("  graphs OFF -> cold ≈ warm (steady-state decode, ~110-118 tok/s)")
    print("  graphs ON  -> cold << warm (cold = capture penalty ~31 tok/s,")
    print("                            warm = replay benefit ~140+ tok/s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
