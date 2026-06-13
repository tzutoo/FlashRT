# FlashRT Qwen3.6-27B on RTX 5090 — Build & Setup Notes

Reproducible guide for running Qwen3.6-27B NVFP4 inference with speculative decode
on a single RTX 5090 inside WSL2 + Docker Desktop, integrated as an opencode LLM backend.

## Step 0: Repository Setup (Fork Workflow)

This is a **fork-tracking-upstream** workflow:

- **Upstream:** `https://github.com/LiangSu8899/FlashRT`
- **Your fork:** `https://github.com/tzutoo/FlashRT`

### Initial clone

```bash
git clone https://github.com/tzutoo/FlashRT.git
cd FlashRT
git remote add upstream https://github.com/LiangSu8899/FlashRT.git
```

### Daily sync with upstream

All local changes (cancel-on-disconnect, developer-role normalize)
are **committed** to the fork. Syncing with upstream is a standard merge:

```bash
git fetch upstream
git merge upstream/main
git push origin main
```

The changes are small and self-contained, so conflicts with upstream are rare.
If upstream touches the same regions of `service.py` or `qwen36_engine.py`,
resolve manually — the intent of each change is documented in Step 2.1.

### .gitignore additions

```gitignore
qwen36_fp8/
qwen36_nvfp4/
opencode.json
```

Model dirs are 29 GB + 26 GB; `opencode.json` contains personal API keys.

## System Requirements

| Component | Spec | Notes |
|-----------|------|-------|
| OS | Windows 11 Pro + WSL2 | Docker Desktop for Windows with WSL2 backend |
| GPU | NVIDIA RTX 5090 (SM120, 32 GB VRAM) | Blackwell; requires CUDA 12.8+ |
| CPU | AMD Ryzen 9 9950X3D (16C/32T) | WSL2 sees 24 threads, 35 GB RAM |
| RAM | 64 GB DDR5-3200 | WSL2 gets ~35 GB by default |
| Disk | ~90 GB free | 30 GB base + 30 GB server + 26 GB NVFP4 + 29 GB FP8 |
| Docker | Desktop 29.x+ with NVIDIA Container Toolkit | GPU passthrough required |

---

## Step 1: Build the Base Docker Image

The base image compiles FlashRT CUDA kernels for SM120 (RTX 5090).

```bash
docker build --progress=plain -t flashrt:5090 \
    -f docker/Dockerfile \
    --build-arg GPU_ARCH=120 \
    --build-arg FA2_ARCH_NATIVE_ONLY=ON \
    --build-arg FA2_HDIMS="128;256" \
    . 2>&1
```

| Setting | Value | Why |
|---------|-------|-----|
| `GPU_ARCH=120` | Target RTX 5090 | Required for Blackwell support |
| `FA2_ARCH_NATIVE_ONLY=ON` | Skip sm_80 + compute_120 PTX | Only native SM120 SASS. ~66% less FA2 compilation |
| `FA2_HDIMS="128;256"` | Restrict FA2 head dims | Only hdim=128 and hdim=256 that Qwen3.6 uses |

Expected build time: **~5 minutes** (282s measured). JOBS defaults to 4 in the
Dockerfile — safe for WSL2's 35 GB RAM (avoids nvcc OOM at `$(nproc)`=24).

Smoke test:
```bash
docker run --rm --gpus all flashrt:5090 \
    python3 -c "import flash_rt; print('flash_rt', flash_rt.__version__); from flash_rt import flash_rt_kernels, flash_rt_fa2; print('kernels + fa2 OK')"
```

---

## Step 2: Build the Server Image

```bash
docker build --progress=plain -t flashrt-server:5090 -f Dockerfile.server . 2>&1
```

`Dockerfile.server`:
```dockerfile
FROM flashrt:5090

RUN pip install --quiet --no-cache-dir fastapi uvicorn 'transformers<4.56'

COPY serving/ /workspace/FlashRT/serving/
WORKDIR /workspace/FlashRT

EXPOSE 8000

HEALTHCHECK --interval=60s --timeout=10s --start-period=90s --retries=2 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)" || exit 1

CMD ["python3", "-m", "serving.qwen36_agent.server", "--checkpoint", "/nvfp4", \
     "--max-seq", "245760", "--route-min-seq", "0", \
     "--default-max-tokens", "8192", "--max-output-tokens", "65536", \
     "--host", "0.0.0.0", "--port", "8000"]
```

### Step 2.1: Local patches (committed to this fork)

| Change | Files | Purpose |
|--------|-------|---------|
| Developer-role normalize | `service.py` | Normalize `developer` → `system`, flatten list-style content blocks |
| Cancel-on-disconnect | `service.py`, `qwen36_engine.py`, `engine.py` | Stop zombie GPU decode after client disconnect |

#### Developer-role normalize

Normalizes `developer` role messages and flattens list-style `content` blocks
that FlashRT's Qwen chat template rejects. Without it, requests from pi fail
with 400 errors. See `docs/pi-developer-role.md` for rationale.

#### Cancel-on-disconnect

See [Client Disconnect / Zombie Process](#client-disconnect--zombie-process) below.

#### Re-applying after upstream conflict

```bash
git diff HEAD...upstream/main -- serving/qwen36_agent/service.py
cat docs/pi-developer-role.patch  # reference for validate_messages() change
```

---

## Step 3: Download Model Checkpoints

Two checkpoints — NVFP4 for the main model and FP8 for the MTP head.

```bash
pip install --user --break-system-packages huggingface_hub
export PATH="$HOME/.local/bin:$PATH"

hf download prithivMLmods/Qwen3.6-27B-NVFP4 --local-dir ./qwen36_nvfp4  # ~26 GB
hf download Qwen/Qwen3.6-27B-FP8 --local-dir ./qwen36_fp8                # ~29 GB
```

| Checkpoint | Path | Purpose |
|------------|------|---------|
| NVFP4 | `./qwen36_nvfp4/` | Main model weights |
| FP8 | `./qwen36_fp8/` | `mtp.safetensors` for speculative decode (K=4→110 tok/s vs K=1→36 tok/s) |

---

## Step 4: Start the Server Container

```bash
docker run --restart always --gpus all --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -p 8000:8000 -d \
    --name flashrt-qwen36 \
    -v $(pwd)/qwen36_nvfp4:/nvfp4:ro \
    -v $(pwd)/qwen36_fp8:/fp8:ro \
    -e FLASHRT_QWEN36_MTP_CKPT_DIR=/fp8 \
    -e FLASHRT_QWEN36_LONG_KV_CACHE=fp8 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    flashrt-server:5090
```

The Dockerfile bakes in:
- `--max-seq 245760` (240K context window)
- `--route-min-seq 0` (FP8-KV path for all prompts — avoids per-position graph capture)
- `--default-max-tokens 8192` (default output budget)
- `--max-output-tokens 65536` (hard cap; input + output share the 240K pool dynamically)

| Env var | Purpose |
|---------|---------|
| `MTP_CKPT_DIR=/fp8` | MTP head location (speculative decode) |
| `LONG_KV_CACHE=fp8` | FP8 KV cache for long context |
| `HF_HUB_OFFLINE=1` | Don't phone home to HuggingFace |

Startup takes ~10s. Verify:

```bash
curl -s http://localhost:8000/v1/models/qwen36-27b
# {"id":"qwen36-27b","object":"model","owned_by":"flash-rt","context_length":245760,"max_output_tokens":65536}

curl -s http://localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"qwen36-27b","messages":[{"role":"user","content":"hi"}],"max_tokens":5}'
```

### Management

```bash
docker logs flashrt-qwen36          # View logs
docker stop flashrt-qwen36          # Stop server
docker start flashrt-qwen36         # Start again
docker rm -f flashrt-qwen36         # Remove container
```

---

## Step 5: Configure opencode

Place in `opencode.json` in your project root (or `~/.config/opencode/opencode.json`):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "flashrt/qwen36-27b",
  "provider": {
    "flashrt": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "FlashRT (local RTX 5090)",
      "options": {
        "baseURL": "http://127.0.0.1:8000/v1",
        "apiKey": "-",
        "headers": { "Authorization": "Bearer -" }
      },
      "models": {
        "qwen36-27b": {
          "name": "Qwen3.6 27B NVFP4 (local RTX 5090)",
          "tool_call": true,
          "temperature": false,
          "limit": { "context": 245760, "output": 65536 },
          "cost": { "input": 0, "output": 0 }
        }
      }
    }
  }
}
```

| Field | Value | Why |
|-------|-------|-----|
| Provider name | `flashrt` (custom) | Built-in `openai` ignores `baseURL` |
| `npm` | `@ai-sdk/openai-compatible` | Required for custom endpoints |
| `temperature` | `false` | Server is greedy-only (no sampling) |
| `baseURL` | `http://127.0.0.1:8000/v1` | Must include `/v1`. Server binds `0.0.0.0` for WSL2 |

---

## Performance

All numbers measured on RTX 5090 (SM120, 32 GB VRAM) in WSL2 Docker,
`--max-seq 245760`, `--route-min-seq 0`, FP8-KV, speculative decode K=4,
CUDA graph flags off (production default).

| Metric | Value |
|--------|-------|
| Cold startup | ~10s |
| Short-prompt decode | **~110 tok/s** |
| 64K context decode | **~89 tok/s** |
| 200K context decode | **~97 tok/s** |
| Without speculative decode | ~36 tok/s (K=1, no MTP) |
| Context window | 240K tokens |
| Max output tokens | 65,536 |
| VRAM usage | ~31.2 GB (leaving ~900 MiB free) |

### Context sweep (128 output tokens, no session reuse)

| Context | TTFT | Decode | Wall |
|---------|------|--------|------|
| ~1K | 51ms | 110 tok/s | 0.8s |
| ~4K | 121ms | 99 tok/s | 1.0s |
| ~16K | 366ms | 91 tok/s | 1.2s |
| ~32K | 673ms | 88 tok/s | 1.8s |
| ~64K | 1.4s | 89 tok/s | 2.5s |
| ~128K | 3.4s | 97 tok/s | 4.1s |
| ~200K | 6.1s | 97 tok/s | 7.1s |

### Multi-turn session (KV cache reuse)

| Turn | Cached | New | Decode |
|------|--------|-----|--------|
| 1 | 0 | 55 | 87 tok/s |
| 2 | 62 | 42 | 81 tok/s |
| 3 | 111 | 42 | 80 tok/s |
| 4 | 160 | 42 | 123 tok/s |
| 5 | 209 | 42 | 86 tok/s |

Only new tokens prefilled each turn. TTFT stays flat (~50ms) regardless of context growth.

### Why 240K instead of 256K?

`--max-seq 262208` (256K) leaves only ~167 MiB VRAM free. At 64K+ real context
this causes catastrophic slowdown (18-21 tok/s) — the GPU is memory-starved
during attention. `--max-seq 245760` (240K) leaves ~900 MiB free, giving
consistent 88-110 tok/s across all context sizes. Tradeoff: 16K less max-input
(180K vs 197K when output is maxed at 65K).

### Speculative decode K

| K | Best for | Decode tok/s |
|---|----------|-------------|
| 4 | Tool-call coding agent (default) | ~110 |
| 3 | Conservative / long plain-text | ~119 |
| 5 | Mixed workloads | ~124 |
| 6 | Fixed-shape benchmarks only | ~134 |

### CUDA Graph replay — why it's off by default

CUDA graphs capture the decode kernel sequence per `cur_pos` and replay it
on subsequent visits to the same position. Measured impact:

| Mode | Cold hit | Warm | Use case |
|------|----------|------|----------|
| Graphs ON (`TQ_VERIFY_GRAPH=1`) | 31 tok/s | **142 tok/s** | Benchmarks, demos |
| Graphs OFF (production default) | **118 tok/s** | **118 tok/s** | Agent serving |

The 20% gap (142 vs 118) comes from eliminating kernel launch overhead
(~0.5ms per decode step) via graph replay.

**Why graphs stay off for agent serving:** every decode step in a growing
session is at a new `cur_pos` — a unique graph key that has never been
visited. The position never repeats, so the graph never warms:

```
Agent session (positions never repeat):
  Request 1: decode pos 34..161  → all cold captures (31 tok/s)
  Request 2: decode pos 200..327 → all cold captures (31 tok/s)
  Request 3: decode pos 360..487 → all cold captures (31 tok/s)

Benchmark (same prompt repeated):
  Run 1: decode pos 34..161 → cold captures (31 tok/s)
  Run 2: decode pos 34..161 → all warm (142 tok/s) ✓
```

For agent serving, graphs add capture cost without benefit. The 118 tok/s
production speed is consistent — no cold hits, no variance.

You can enable graphs for benchmark/demo use:
```bash
docker run ... -e FLASHRT_QWEN36_TQ_VERIFY_GRAPH=1 \
    -e FLASHRT_QWEN36_TQ_MTP_CHAIN_GRAPH=1 \
    flashrt-server:5090 --max-seq 32768 --route-min-seq 0
```

---

## Upstream Features Merged (2026-06-13)

| Feature | What it does | Benefit |
|---------|-------------|---------|
| Qwen3.6 Spark support | SM121 frontend (GB10/DGX Spark) | Future Spark GPU support |
| SM121 runtime defaults | SM120 fastgemm only on `(12,0)` | Prevents slow kernels on Spark |
| Pi0.5 split-KV decoder | Joint-attention in fixed-shape mode | 3× faster if running Pi0.5 |
| Pi0.5 autotune idempotent | `_gemms_autotuned` guard | Eliminates redundant cuBLASLt benchmarking |
| FA2 wrapper + bindings | Native FA2 for Spark | SM121 attention path |
| CMakeLists SM121 support | `GPU_ARCH=121` recognized | Build for Spark hardware |

---

## Docker Images

| Image | Size | Contents |
|-------|------|----------|
| `flashrt:5090` | 29.7 GB | PyTorch 2.9 + CUDA 13.0 + FlashRT kernels (SM120) |
| `flashrt-server:5090` | 29.9 GB | Base + fastapi + uvicorn + transformers + serving code |

---

## Known Limitations

| Limitation | Details |
|------------|---------|
| Greedy decode only | No temperature/top_p/top_k. Server accepts but ignores sampling params. |
| Single tool call per turn | Server stops on first complete tool call. |
| NVFP4 hardcoded | Switching to FP8 requires engine + kernel changes. |
| Thinking mode disabled | `enable_thinking` defaults to false. |

---

## File Map

```
FlashRT/
├── Dockerfile.server              # Server image (liveness healthcheck + CMD)
├── opencode.json                  # opencode config (project-local)
├── qwen36_nvfp4/                  # NVFP4 main model (~26 GB)
├── qwen36_fp8/                    # FP8 with mtp.safetensors (~29 GB)
├── serving/qwen36_agent/
│   ├── server.py                  # FastAPI server (/v1/models, /health)
│   ├── service.py                 # [patched] cancel-on-disconnect + developer-role normalize
│   ├── engine.py                  # [patched] generate_stream() cancel param
│   ├── qwen36_engine.py           # [patched] cancel-aware _committed_stream
│   └── tool_stream.py             # Tool call XML/JSON parser
├── docs/
│   ├── pi-developer-role.md       # Developer-role patch rationale
│   └── pi-developer-role.patch    # Patch file (validate_messages only)
├── flash_rt/
│   ├── flash_rt_kernels.cpython-312-x86_64-linux-gnu.so
│   └── flash_rt_fa2.cpython-312-x86_64-linux-gnu.so
└── BUILD_NOTES.md                 # This file
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Build hangs for hours | `FA2_ARCH_NATIVE_ONLY=ON` must be set. JOBS defaults to 4 (safe for 35 GB RAM) |
| `ImportError: flash_rt_kernels` | Rebuild base image |
| opencode says "Not Found" | Provider must be custom name (not `openai`), must include `npm` |
| Server unreachable | Container must bind `0.0.0.0`. `baseURL` must include `/v1` |
| Client gets no response | See [Client Disconnect](#client-disconnect--zombie-process) below |

---

## Client Disconnect / Zombie Process

### The problem

Without the cancel-on-disconnect fix, the streaming path holds a single-thread
lock for the entire generation. When the client disconnects, GPU decode
continues until `max_tokens` is exhausted — a zombie. The lock blocks all
subsequent requests until the zombie finishes, causing a cascade:

```
Request A starts → client times out → zombie A holds lock
Request B arrives → blocks → client B times out → zombie B queues ...
```

### The fix

`stream_openai()` in `service.py` uses a **producer-consumer queue** so SSE
yields happen *outside* the GPU lock:

1. A daemon thread runs prefill + decode under `self._lock`, pushing chunks into a bounded `queue.Queue(maxsize=4)`.
2. The outer generator reads from the queue and yields SSE events — no lock held during I/O.
3. A `threading.Event` (`cancel`) is threaded through to `qwen36_engine.py`.

**Cancellation chain:**

1. When the consumer dies (client disconnect), `queue.put()` starts timing out (0.5s per attempt).
2. After 5 consecutive full-queue timeouts (2.5s), the producer infers consumer death → `cancel.set()`.
3. `_committed_stream()` in `qwen36_engine.py` checks `cancel.is_set()` between decode chunks and returns early.
4. Lock is released. Total detection + cancel: ~5s.

This avoids relying on `GeneratorExit` (which Starlette's `StreamingResponse`
never fires for SSE streams).

---

## GPU Health Monitoring

The healthcheck uses `GET /health` — returns `{"status":"ok"}` without GPU
inference. Zero VRAM overhead, zero interference with active requests. If the
server crashes (CUDA error, OOM, segfault), Docker's `--restart always`
restarts automatically.

| Parameter | Value | Why |
|-----------|-------|-----|
| `--interval` | 60s | Standard cadence |
| `--timeout` | 10s | `/health` is instant |
| `--start-period` | 90s | Model load + margin |
| `--retries` | 2 | One transient failure tolerated |
