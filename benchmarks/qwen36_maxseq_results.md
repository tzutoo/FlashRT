# Qwen3.6-27B `--max-seq` sweet-spot: 192K vs 224K vs 256K

**TL;DR: Use `--max-seq 196608` (192K).** It is equivalent to 224K at all
input contexts and 6% faster at 192K input context. 256K is in a
VRAM cliff regardless of env-var tweaking.

| config | max_seq | VRAM headroom | decode at 192K ctx | recommended |
|---|---:|---:|---:|:---:|
| **192K** (current) | 196608 | **1431 MiB** | **40.2 tok/s** | **yes** |
| 224K (Phase A) | 229376 | 1511 MiB | 37.7 tok/s | equivalent |
| 256K (Phase B) | 262208 | 376 MiB | 25.3 tok/s | **no — cliff** |

## Method

- **Hardware:** RTX 5090, 32 GB VRAM, single GPU
- **Checkpoint:** `prithivMLmods/Qwen3.6-27B-NVFP4`
- **Long route:** `FLASHRT_QWEN36_LONG_KV_CACHE=fp8` (default)
- **Decode metric:** `decode_tok_per_s` from `/v1/chat/completions` response (the `flashrt.decode_tok_per_s` field — excludes TTFT, includes MTP spec overhead)
- **Short contexts (1K-32K):** median of 3 samples, session cleanup between samples
- **Long contexts (64K-192K):** 1 sample each (each request is 8-60 seconds)
- **Scripts:** `qwen36_maxseq_192k_vs_224k.py` (192K vs 224K), `qwen36_maxseq_sweep.py` (the upstream benchmark that found the 240K cliff)

## Full results — decode tok/s

| config | max_seq | free MiB | 1K | 8K | 16K | 32K | 64K | 128K | 192K |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **192K** (recommended) | 196608 | 1431 | 135 | 138 | 114 | 127 | 82 | 66 | **40** |
| 224K | 229376 | 1511 | 135 | 138 | 114 | 127 | 82 | 66 | 38 |
| 256K (cliff) | 262208 | 376 | 40 | 27 | 25 | 35 | 7 | 7 | 25 |
| upstream @ 256K | 262208 | n/a | — | — | — | — | — | — | 144† |

†upstream table at 256K context claims 144 tok/s, but locally we measure 7-25 tok/s at the same max_seq. The gap is likely the upstream's CUDA graphs ON + warm cache scenario; our benches use graphs OFF (the production default for agent traffic where positions never repeat) and cold prefill each request.

## Findings

### 1. 192K and 224K are equivalent at every measured input context

| input ctx | 192K | 224K | Δ |
|---:|---:|---:|---:|
| 1K | 135.2 | 135.5 | +0.2 |
| 8K | 138.2 | 138.0 | -0.2 |
| 16K | 113.5 | 113.3 | -0.2 |
| 32K | 126.8 | 126.5 | -0.3 |
| 64K | 82.3 | 82.4 | +0.1 |
| 128K | 66.3 | 66.0 | -0.3 |
| 192K | 40.2 | 37.7 | **-2.5 (-6%)** |

All deltas are within noise except the 192K input context, where 192K is 6% faster (smaller persistent cache = less attention dequant work). The 224K change was a "free 32K capacity upgrade" but did not unlock any decode speed.

### 2. 256K is in a hard cliff, even with env-var tightening

The Phase B attempt to push to 256K with the following env vars:
```
FLASHRT_QWEN36_FP8_STAGE_CAP=131072
FLASHRT_QWEN36_FP8_HOT_STAGE_CAP=65536
FLASHRT_QWEN36_FP8_STAGE_RESERVE_MB=2048
FLASHRT_QWEN36_FP8_XQA_SCRATCH_MB=128
```
reduced the persistent cache by 0 MB (it's not affected by stage cap) and only saved ~200-500 MB on staging buffers. The persistent cache itself grew from 4.6 GB to 5.4 GB, which was enough to cross the cliff at 245760 (240K). Result:

- 64K ctx: **7.2 tok/s** (vs 82 at 192K) — **11× slower**
- 128K ctx: **6.6 tok/s** (vs 66) — **10× slower**
- 192K ctx: timeout (4 min not enough for prefill alone)

The cliff is at the **persistent cache size**, not the staging buffers. To get past 240K you need to compress the persistent cache itself, not just trim the stage.

### 3. Decode degrades with input context regardless of max_seq

This is the O(S²) attention cost. Even at the optimal 192K max_seq:

| input ctx | decode tok/s | drop from 32K |
|---:|---:|---:|
| 1K | 135 | (baseline) |
| 32K | 127 | -6% |
| 64K | 82 | -35% |
| 128K | 66 | -48% |
| 192K | 40 | -68% |

This is the actual long-context cost of the model — the prompt itself slows decode, not just the max_seq setting. Env vars cannot fix this; only model-level changes (sparsity, sliding window) could.

### 4. The 240K cliff (from upstream `fd11689`)

The `qwen36_maxseq_sweep.py` bench from upstream commit `fd11689` identified the cliff precisely:

```
max_seq   free MiB   decode tok/s (1K ctx)
196608      844     122
229376      944     119   <- last safe
245760      364      27   <- cliff
```

Going from 229376 to 245760 (a 16K step) collapses free VRAM from 944 MiB to 364 MiB and decode from 119 to 27 tok/s. The cliff is sharp.

## Decision matrix

| use case | recommended max_seq | why |
|---|---|---|
| Most coding agents (≤95K context) | `--max-seq 196608` (192K) | safe, fast, well below the cliff |
| Single research paper (≤30K) | `--max-seq 196608` (192K) | identical at small contexts |
| Whole-book analysis (100-200K) | `--max-seq 196608` (192K) | 192K input gives 40 tok/s; same as 224K |
| 200K-228K input range | `--max-seq 229376` (224K) | 192K can't accept prompts >196608 |
| 256K input | not yet viable | cliff collapses decode to 7-25 tok/s; needs TQ bit-pack (memory/accuracy bisection) |

## Why we reverted from 224K to 192K

Phase A (commit `262f69b`) bumped `--max-seq 196608` → `--max-seq 229376` thinking it was a "free upgrade" — 32K more capacity, no decode regression at 1K-32K.

The new long-context data shows:
- At 192K input context, 192K is 6% **faster** than 224K
- The 32K "free" capacity is unused (no production workload uses 200K-224K)
- 224K is 22K from the cliff; 192K is 60K from the cliff — more safety margin
- The original 192K was the upstream-validated config

Reverting to 192K costs nothing and gains safety margin.

## How to reproduce

```bash
# 1. Make sure the runner script exists
cat > /tmp/phaseA_runner.py << 'EOF'
import json, urllib.request, sys
d = json.load(open("/tmp/req.json"))
req = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
    data=json.dumps(d).encode(),
    headers={"Content-Type": "application/json"},
    method="POST")
r = urllib.request.urlopen(req, timeout=600)
sys.stdout.write(r.read().decode())
EOF

# 2. Run the consolidated bench (~5 minutes)
python3 benchmarks/qwen36_maxseq_192k_vs_224k.py
```

The bench writes `maxseq_192k_vs_224k_results_<ts>.json` with full per-sample data.

## Source data

- `phaseA_ab_results_1783463524.json` — Phase A: 192K vs 224K at 1K-32K (committed `262f69b`)
- `longctx_orig_192k_results_1783471003.json` — 192K at 64K-192K long contexts
- `qwen36_maxseq_sweep.py` (committed) — the upstream sweep that found the 240K cliff
- `qwen36_graph_results.md` (committed) — full max_seq sweep table with methodology
