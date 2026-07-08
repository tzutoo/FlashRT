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

## Root cause found (2026-06-18, follow-up)

**Both open questions are one bug: `--max-seq 245760` oversizes the
long-context KV cache and memory-starves the card.** Single-variable proof —
identical graphs-OFF config, identical prompts, only `--max-seq` differs:

| metric | `--max-seq 245760` | `--max-seq 32768` |
|---|---|---|
| idle VRAM used | ~31 880 MiB (~700 MiB free) | ~25 085 MiB (~7.5 GB free) |
| decode @1K ctx | 120 tok/s | 130 tok/s |
| decode @4K–32K ctx | **~24 tok/s** (cliff at the 2048 BF16 window) | **~130 tok/s** (flat) |
| 32K request decode | 19 tok/s | **95 tok/s** |
| short prompt *after* a 32K req | **21 tok/s** (stuck slow) | **104 tok/s** (no slowdown) |

Mechanism: in long-ctx mode the BF16 spec window is `min(2048, MAX_Q_SEQ)`.
With `--route-min-seq 0`, *every* prompt (even 38 tokens) takes the long
TQ-packed route. At `max_seq=245760` the TQ cache is sized for 240K → ~700 MiB
free → attention is memory-starved → decode collapses to ~24 tok/s for anything
above the 2048 window, **and** the long path gets stuck slow even for short
prompts after any long request (deleting sessions does not help; only a restart
restores speed — the frontend GPU state, not the session registry, holds the
slowdown).

At `max_seq=32768` the cache is sized for 32K → ~7.5 GB free → decode stays
~130 tok/s flat across 1K–28K and short prompts stay fast after long requests.

## FINAL: default is `--max-seq 196608` (192K) (2026-06-18)

The earlier 131072 (128K) recommendation was too conservative for real
pi/coding-agent traffic that pastes whole articles or large files — those
sessions routinely hit 95K tokens, and at 128K they overflowed `max_seq`
mid-request, crashing the SSE stream (`Stream ended without finish_reason`).

`--max-seq 196608` (192K) fits a 95K session + multi-turn growth + the 8192
default output budget, with measured steady-state decode ~108–120 tok/s up to
32K real context. It leaves ~1.3 GB idle free VRAM (borderline on WSL2 — under
extreme session growth + unlucky free-VRAM drift you may see ~30 tok/s; a
`docker restart` clears it).

Two guards make 192K safe as the default:
1. **Overflow no longer crashes the stream** — `service._reclip_max_tokens_for_engine`
   re-clips the output budget against the actual engine token count (the hot
   session journal, longer than the rendered prompt because of hidden Qwen3.6
   control tokens); if a session truly fills `max_seq` it returns a clean HTTP
   400 instead of a broken SSE stream.
2. **`docker stop` releases VRAM deterministically** (the `release_gpu()` work),
   so recovery from the slow state is a ~10 s stop/start.

For sessions that stay small and want bulletproof decode speed with no
restarts, `--max-seq 131072` (128K, ~3.5 GB free) remains a valid conservative
choice — verified with `qwen36_pi_growth_sim.py` (49K session, ~80 tok/s, free
VRAM steady, no restart):

| turn | context | decode | free VRAM |
|---:|---:|---:|---:|
| 1  | 11K | 89 tok/s | 3083 MiB |
| 14 | 49K | 79 tok/s | 3083 MiB |

### Why graphs OFF stays correct at every max_seq

CUDA graphs are keyed on the **absolute decode position** (`cur_pos`). An agent
conversation grows monotonically, so every generated token is at a
never-visited position → under graphs ON every decode step is a ~23 tok/s cold
capture, never a warm replay. `max_seq` doesn't change this — the decision is
about whether positions *repeat*, not the window size. Graphs ON only wins when
the *same* prompt re-runs (benchmarks/demos). Confirmed by `qwen36_agent_sim.py`:

| turn | graphs OFF @32k | graphs ON @32k |
|---|---|---|
| 1 | 106 tok/s | 31 tok/s |
| 5 |  87 tok/s | 21 tok/s |

The serving README explicitly says the same: "Do not set
`FLASHRT_QWEN36_TQ_VERIFY_GRAPH=1`... for normal agent serving."

### Recommendation

Run with `--max-seq 196608` (192K) — fits paste-heavy/article sessions (95K+)
with multi-turn growth, ~108–120 tok/s steady-state, with the overflow-reclip
fix making the edge case return a clean 400 instead of crashing the stream.
This is the committed default in `Dockerfile.server`. For small-session
workloads that prioritize bulletproof decode speed, `--max-seq 131072` (128K,
~3.5 GB free, no restarts) is the conservative alternative.

```bash
docker run ... flashrt-server:5090 \
    python3 -m serving.qwen36_agent.server --checkpoint /nvfp4 \
    --max-seq 196608 --route-min-seq 0 \
    --default-max-tokens 8192 --max-output-tokens 65536 \
    --host 0.0.0.0 --port 8000
```

### Methodology lesson (important)

Single-sample sweeps are misleading here. An earlier single-sample sweep
reported 116 tok/s at 16K for `245760` — a lucky first sample before the TQ
cache filled. The 8-sample re-measurement showed steady ~31 tok/s. Always use
multiple samples + session cleanup between them (`qwen36_maxseq_sweep.py`).

## Graphs ON even at 32768? No — agent positions never repeat (2026-06-18)

A natural follow-up: once `--max-seq 32768` removes the VRAM starvation, do
graphs ON start helping? **No.** The graph decision depends on whether decode
*positions repeat*, not on `max_seq`. An agent conversation grows monotonically,
so every generated token is at a never-visited `cur_pos` → every decode step is
a cold capture, never a warm replay. Verified with `qwen36_agent_sim.py`, a
growing-conversation simulation (one session, history appended each turn):

| turn | graphs OFF @32k | graphs ON @32k |
|---|---|---|
| 1 | 106 tok/s | 31 tok/s |
| 2 |  98 tok/s | 24 tok/s |
| 3 |  87 tok/s | 23 tok/s |
| 4 |  94 tok/s | 23 tok/s |
| 5 |  87 tok/s | 21 tok/s |

Graphs ON is 3–4× slower for agent traffic and even degrades slightly over
turns (cache fills with captures that never replay; eviction adds overhead).
Graphs OFF is steady ~90 tok/s.

**Conclusion: graphs OFF is correct for pi/agent serving at every `max_seq`.**
Graphs ON only wins when the *same* prompt re-runs (benchmarks, demos, prompt
replay) — `qwen36_graph_sweep.py`'s warm column.

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
python3 benchmarks/qwen36_graph_sweep.py --url http://127.0.0.1:8765 \
    --contexts 1024 4096 16384 32768 --out-tokens 200
```
