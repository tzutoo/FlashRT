# Qwen3.6 CUDA Graph ON vs OFF — context sweep (RTX 5090, 32 GB)

Measured 2026-06-18 with `qwen36_graph_sweep.py`. Same hardware/software as
`BUILD_NOTES.md` (Qwen3.6-27B NVFP4, MTP spec decode K=4, FP8-KV, RTX 5090,
WSL2 + Docker Desktop).

## Method

For each context size the script sends two **independent** requests that share
an **identical prompt**, with a fresh `flashrt_session_id` + unique
`flashrt_cache_salt` each call (so prefix/KV cache never reuses → both runs
cold-prefill). The CUDA-graph cache is **global** to the frontend, so run 2
revisits the same decode positions → warm replay. Sessions are DELETEd after
each request so a long sweep doesn't accumulate KV and starve VRAM (see
[Open questions](#open-questions)).

- **cold** = first visit to these decode positions (graph capture if graphs ON)
- **warm** = second visit (graph replay if graphs ON)

## Results

### Graphs OFF (production default, `--max-seq 245760`, `verify_graph=0`)

```
   ctx  prompt   out  cold tok/s  warm tok/s  cold prefill ms
   1024     980   200       120.9       122.2               98
   4096    3710   200        30.2        28.7              679
  16384   14631   200        26.1        25.9             3887
  32768   29191   200        26.6        25.8             8785
```

### Graphs ON (`--max-seq 32768`, `verify_graph=1 mtp_chain_graph=1`)

```
   ctx  prompt   out  cold tok/s  warm tok/s  cold prefill ms
   1024     980   200        32.4       141.2               36
   4096    3710   200        33.0       141.5              439
  16384   14631   200        32.9       136.0             1342
  28672   25551   200        38.3       149.7             2589
```

## Reading it

| | cold | warm |
|---|---|---|
| graphs OFF | 120 tok/s | 122 tok/s |
| graphs ON | **32 tok/s** (capture) | **141 tok/s** (replay) |

- **graphs OFF** is steady-state ~120 tok/s at 1K (matches `BUILD_NOTES.md`).
  For agent serving (pi), where every generated token is a new, never-repeated
  decode position, OFF is the right choice: ON would drop every step to the
  ~32 tok/s capture path.
- **graphs ON warm** (~141 tok/s) beats OFF, and stays **flat** across
  1K→28K. It only wins when the *same* decode positions recur (benchmarks,
  fixed-shape demos, prompt replay).
- **graphs ON cold** (~32 tok/s) is the per-position capture cost — the exact
  number `BUILD_NOTES.md` cites.

## Open questions (see commit history / follow-up)

1. **graphs-OFF ≥4K slowdown.** OFF collapses to ~26 tok/s at 4K–32K and
   plateaus, vs `BUILD_NOTES.md`'s ~88–99. Prefill is also ~3.4× slower
   (8.8s vs 2.6s at ~30K). The graphs-ON run used `--max-seq 32768` while
   OFF used `--max-seq 245760`; the long-ctx-mode allocation keyed on
   `max_seq` is the prime suspect — not a routing switch (both use the long
   route with `route_min=0`).
2. **KV-churn degradation.** Hammering the server with many fresh large-context
   sessions (no session cleanup) collapsed decode from ~96 → ~17 tok/s until a
   container restart. The serving layer's session/KV reclamation under churn
   does not gracefully reclaim VRAM.

Both are real and separate from the graph on/off decision.

## How to reproduce

```bash
# Graphs OFF (production) — the running container is already this:
docker run ... flashrt-server:5090

# Graphs ON:
docker run ... -e FLASHRT_QWEN36_TQ_VERIFY_GRAPH=1 \
    -e FLASHRT_QWEN36_TQ_MTP_CHAIN_GRAPH=1 \
    flashrt-server:5090 \
    python3 -m serving.qwen36_agent.server --checkpoint /nvfp4 \
    --max-seq 32768 --route-min-seq 0 \
    --default-max-tokens 8192 --max-output-tokens 65536 \
    --host 0.0.0.0 --port 8000

# Sweep:
python3 benchmarks/qwen36_graph_sweep.py --url http://127.0.0.1:8000 \
    --contexts 1024 4096 16384 32768 --out-tokens 200
```
