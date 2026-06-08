# FlashRT Qwen3.6-27B on RTX 5090 — Build & Setup Notes

Reproducible guide for running Qwen3.6-27B NVFP4 inference with speculative decode
on a single RTX 5090 inside WSL2 + Docker Desktop, integrated as an opencode LLM backend.

## Step 0: Repository Setup (Fork Workflow)

This is a **fork-tracking-upstream** workflow. The setup assumes:

- **Upstream:** `https://github.com/LiangSu8899/FlashRT` (the original)
- **Your fork:** `https://github.com/tzutoo/FlashRT` (where you push your changes)

### Initial clone

```bash
# Clone YOUR fork (not the upstream)
git clone https://github.com/tzutoo/FlashRT.git
cd FlashRT

# Add upstream as a separate remote
git remote add upstream https://github.com/LiangSu8899/FlashRT.git
git remote -v
#   origin    https://github.com/tzutoo/FlashRT.git (fetch)
#   origin    https://github.com/tzutoo/FlashRT.git (push)
#   upstream  https://github.com/LiangSu8899/FlashRT.git (fetch)
#   upstream  https://github.com/LiangSu8899/FlashRT.git (push)
```

> If you cloned the upstream directly (like the original `git clone
> https://github.com/flash-rt/FlashRT.git`), rename and add your fork:
>
> ```bash
> git remote rename origin upstream
> git remote add origin https://github.com/tzutoo/FlashRT.git
> ```

### Daily sync with upstream

All local changes (cancel-on-disconnect, healthcheck, developer-role normalize)
are **committed** to the fork. Syncing with upstream is a standard merge:

```bash
# 1. Fetch and merge upstream
git fetch upstream
git merge upstream/main
# Resolve any conflicts in the modified files, then commit.

# 2. Push to your fork (keeps fork in sync)
git push origin main
```

The changes are small and self-contained, so conflicts with upstream are rare.
If upstream touches the same regions of `service.py` or `qwen36_engine.py`,
resolve manually — the intent of each change is documented in Step 2.1.

### .gitignore additions for this fork

Add these lines to `.gitignore` (they are repo-specific and not safe to commit):

```gitignore
# Model checkpoints (too large for git)
qwen36_fp8/
qwen36_nvfp4/

# Personal opencode config (contains API keys)
opencode.json
```

The model dirs are 29 GB + 26 GB; `opencode.json` contains your personal API
keys (Z.AI vision, NanoGPT) and should not be public.

## System Requirements

| Component | Spec | Notes |
|-----------|------|-------|
| OS | Windows 11 Pro + WSL2 | Docker Desktop for Windows with WSL2 backend |
| GPU | NVIDIA RTX 5090 (SM120, 32 GB VRAM) | Blackwell; requires CUDA 12.8+ |
| CPU | AMD Ryzen 9 9950X3D (16C/32T) | WSL2 sees 24 threads, 35 GB RAM (auto-assigned by Docker Desktop) |
| RAM | 64 GB DDR5-3200 | WSL2 gets ~35 GB by default (Docker Desktop auto-allocation) |
| Disk | ~90 GB free | 30 GB base image + 30 GB server image + 26 GB NVFP4 ckpt + 29 GB FP8 ckpt |
| Docker | Desktop 29.x+ with NVIDIA Container Toolkit | GPU passthrough required |

> **WSL2 resource note:** WSL2 on Windows 11 auto-assigns half of system RAM (35 GB of 64 GB)
> and all logical processors (24 of 32 threads visible to Docker). You can override in
> `%USERPROFILE%\.wslconfig` if needed.

---

## Step 1: Build the Base Docker Image

The base image compiles FlashRT CUDA kernels for SM120 (RTX 5090).

```bash
# Clone the repo
git clone https://github.com/flash-rt/FlashRT.git
cd FlashRT

# Build — must cap parallelism to avoid OOM during nvcc compilation
docker build --progress=plain -t flashrt:5090 \
    -f docker/Dockerfile \
    --build-arg GPU_ARCH=120 \
    --build-arg FA2_ARCH_NATIVE_ONLY=ON \
    --build-arg FA2_HDIMS="128;256" \
    . 2>&1
```

### Build Notes

| Setting | Value | Why |
|---------|-------|-----|
| `FA2_ARCH_NATIVE_ONLY=ON` | Skip sm_80 + compute_120 PTX | Only builds native SM120 SASS for FlashAttention2. ~66% less FA2 compilation work |
| `FA2_HDIMS="128;256"` | Restrict FA2 head dims | Only compile the hdim=128 and hdim=256 kernels Qwen3.6 uses. Skips hdim=96 |
| `GPU_ARCH=120` | Target RTX 5090 | Required for Blackwell support |

Expected build time: **~5 minutes** for CUDA kernel compilation (282s measured).

Smoke test inside the image:
```bash
docker run --rm --gpus all flashrt:5090 \
    python3 -c "import flash_rt; print('flash_rt', flash_rt.__version__); from flash_rt import flash_rt_kernels, flash_rt_fa2; print('kernels + fa2 OK')"
```

Expected output: `flash_rt 0.1.0` + `kernels + fa2 OK`

---

## Step 2: Build the Server Image

Extends the base image with FastAPI, uvicorn, and the serving code baked in.
No pip install at runtime — faster startup.

```bash
# From the FlashRT repo root
docker build --progress=plain -t flashrt-server:5090 -f Dockerfile.server . 2>&1
```

`Dockerfile.server`:
```dockerfile
FROM flashrt:5090

RUN pip install --quiet --no-cache-dir fastapi uvicorn 'transformers<4.56'

COPY serving/ /workspace/FlashRT/serving/
WORKDIR /workspace/FlashRT

EXPOSE 8000

# Health check: sends a tiny completion request. Only kills on severe
# degradation (< 5 tok/s = real GPU corruption). Normal post-large-context
# variance (10-20 tok/s on tiny 5-token outputs) does NOT trigger restart.
COPY <<'EOF' /usr/local/bin/healthcheck.py
import urllib.request, json, sys, os, signal
try:
    req = urllib.request.Request(
        'http://localhost:8000/v1/chat/completions',
        json.dumps({'model': 'qwen36-27b', 'messages': [{'role': 'user', 'content': 'ping'}], 'max_tokens': 5}).encode(),
        headers={'Content-Type': 'application/json'})
    resp = urllib.request.urlopen(req, timeout=15)
    data = json.loads(resp.read())
    tok_s = data.get('flashrt', {}).get('decode_tok_per_s', 0)
    if tok_s > 0 and tok_s < 5:
        print(f'DEGRADED: {tok_s:.1f} tok/s - GPU corrupt, killing server for restart')
        os.kill(1, signal.SIGTERM)
        sys.exit(1)
except Exception as e:
    print(f'HEALTH FAIL: {e} - killing server for restart')
    try:
        os.kill(1, signal.SIGTERM)
    except Exception:
        pass
    sys.exit(1)
EOF

HEALTHCHECK --interval=60s --timeout=15s --start-period=90s --retries=1 \
    CMD python3 /usr/local/bin/healthcheck.py

CMD ["python3", "-m", "serving.qwen36_agent.server", "--checkpoint", "/nvfp4", "--max-seq", "262208", "--default-max-tokens", "8192", "--max-output-tokens", "32768", "--host", "0.0.0.0", "--port", "8000"]
```


### Step 2.1: Custom patches (committed to this fork)

The fork carries **three** sets of local changes on top of upstream, all committed:

| Change | Files | Purpose |
|--------|-------|---------|
| Pi developer-role patch | `service.py` | Normalize `developer` → `system` role, flatten list-style content blocks |
| Cancel-on-disconnect | `service.py`, `qwen36_engine.py`, `engine.py` | Stop zombie GPU decode after client disconnect |
| Decode-speed healthcheck | `Dockerfile.server` | Auto-restart container when GPU is degraded |

These are already committed — just build:

```bash
docker build --progress=plain -t flashrt-server:5090 -f Dockerfile.server . 2>&1
```

#### The pi developer-role patch

The pi developer-role patch (`docs/pi-developer-role.patch`) normalises `developer`
role messages and flattens list-style `content` blocks that FlashRT's Qwen chat
template rejects. Without it, requests from pi fail with 400 errors:
`unsupported role: 'developer'`, `jinja2.exceptions.TemplateError: System message
must be at the beginning.`, or `message.content must be a string`.

See `docs/pi-developer-role.md` for the full rationale.

#### The cancel-on-disconnect mechanism

The cancel-on-disconnect changes (`service.py`, `qwen36_engine.py`, `engine.py`)
prevent zombie GPU decode after a client disconnect. See the
[Client Disconnect / Zombie Process](#client-disconnect--zombie-process) section
for details.

#### The decode-speed healthcheck

The `Dockerfile.server` healthcheck sends a tiny completion request every 60s
and inspects `decode_tok_per_s`. If the GPU is severely degraded (under 5 tok/s,
indicating real corruption from CUDA graph state), the healthcheck kills PID 1 to
trigger a Docker auto-restart. It also kills on any exception (CUDA error, timeout).
The threshold is intentionally conservative: normal post-large-context variance can
cause tiny 5-token outputs to dip to 10-20 tok/s, which is not corruption.
See the [GPU Health Monitoring](#gpu-health-monitoring) section below.

#### Re-applying changes after a conflict with upstream

If `git merge upstream/main` conflicts with the local patches, resolve manually.
The patch files are kept in `docs/` as a reference for the original intent:

```bash
# Check what upstream changed
git diff HEAD...upstream/main -- serving/qwen36_agent/service.py

# The pi developer-role patch is preserved as a reference
cat docs/pi-developer-role.patch
```

The patch file only covers the `validate_messages()` change. The cancel-on-disconnect
and healthcheck changes are larger — see the sections below for their rationale.

---


## Step 3: Download Model Checkpoints

You need **two** checkpoints — NVFP4 for the main model and FP8 for the MTP (speculative decode) head.

```bash
# Install HuggingFace CLI (if not already)
pip install --user --break-system-packages huggingface_hub
export PATH="$HOME/.local/bin:$PATH"

# NVFP4 main model (26 GB)
hf download prithivMLmods/Qwen3.6-27B-NVFP4 --local-dir ./qwen36_nvfp4

# FP8 model — needed for mtp.safetensors (speculative decode)
hf download Qwen/Qwen3.6-27B-FP8 --local-dir ./qwen36_fp8
```

| Checkpoint | Path | Size | Purpose |
|------------|------|------|---------|
| NVFP4 | `./qwen36_nvfp4/` | ~26 GB | Main model weights |
| FP8 | `./qwen36_fp8/` | ~29 GB | Contains `mtp.safetensors` for speculative decode |

> Without `mtp.safetensors`, speculative decode is disabled and you get ~36 tok/s
> instead of ~120+ tok/s.

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
    -e FLASHRT_QWEN36_LONG_CTX_ROUTE_MIN_SEQ=512 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    flashrt-server:5090
```

The Dockerfile bakes in `--max-seq 262208` (256K context), `--default-max-tokens 8192`,
and `--max-output-tokens 32768`. Clients get 8192 output tokens by default; if a client
explicitly requests more (up to 32768), the server allows it. Override at the
`docker run` command line if needed.

| Flag | Purpose |
|------|---------|
| `--restart always` | Auto-starts with Docker Desktop, auto-restarts on crash |
| `--gpus all` | GPU passthrough |
| `--ipc=host` | Shared memory for CUDA |
| `-p 8000:8000` | Expose server port |
| `MTP_CKPT_DIR=/fp8` | MTP head location (speculative decode) |
| `LONG_KV_CACHE=fp8` | FP8 KV cache for 256K context |
| `LONG_CTX_ROUTE_MIN_SEQ=512` | Route prompts ≥512 tokens through chunked FP8-KV |
| `HF_HUB_OFFLINE=1` | Don't phone home to HuggingFace Hub |
| `TRANSFORMERS_OFFLINE=1` | Same for transformers library |
| `/nvfp4:ro` / `/fp8:ro` | Read-only model mounts |

Startup takes ~30 seconds (model loading + MTP head). Verify:

```bash
# Wait for startup, then test
curl -s http://localhost:8000/v1/models/qwen36-27b
# Expected: {"id":"qwen36-27b","object":"model","owned_by":"flash-rt"}

curl -s http://localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"qwen36-27b","messages":[{"role":"user","content":"hi"}],"max_tokens":5}'
```

### Management Commands

```bash
docker logs flashrt-qwen36          # View logs
docker stop flashrt-qwen36          # Stop server
docker start flashrt-qwen36         # Start again (preserves --restart)
docker rm -f flashrt-qwen36         # Remove container
```

---

## Step 5: Configure opencode

Place this in `opencode.json` in your project root (or `~/.config/opencode/opencode.json` for global):

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
        "headers": {
          "Authorization": "Bearer -"
        }
      },
      "models": {
        "qwen36-27b": {
          "name": "Qwen3.6 27B NVFP4 (local RTX 5090)",
          "tool_call": true,
          "temperature": false,
          "limit": {
            "context": 262208,
            "output": 8192
          },
          "cost": {
            "input": 0,
            "output": 0
          }
        }
      }
    }
  }
}
```

### Key config details

| Field | Value | Why |
|-------|-------|-----|
| Provider name | `flashrt` (custom) | Must be a custom name, not `openai`. Built-in `openai` ignores `baseURL` and routes to `api.openai.com` |
| `npm` | `@ai-sdk/openai-compatible` | Required for custom OpenAI-compatible endpoints |
| `apiKey` | `-` | Dummy — server has no auth |
| `temperature` | `false` | Server is greedy-only (no sampling). Tells opencode not to send temperature params |
| `tool_call` | `true` | Enable tool calling support |
| `baseURL` | `http://127.0.0.1:8000/v1` | Must include `/v1` suffix. Server must bind `0.0.0.0` (not `127.0.0.1`) for WSL2 Docker port forwarding |

---

## Performance

| Metric | Value |
|--------|-------|
| Cold startup | ~30s (model load + MTP head) |
| Decode throughput | ~120-134 tok/s (speculative decode K=4-6) |
| Without speculative decode | ~36 tok/s (K=1, no MTP) |
| Context window | 256K tokens (FP8 KV cache) |
| VRAM usage | ~30 GB (model + KV cache + MTP) |

### Speculative decode K recommendation

| Workload | K | Decode tok/s |
|----------|---|-------------|
| Tool-call coding agent (default) | 4 | ~140 |
| Short generations, single prompt class | 6 | ~134 |
| Mixed workloads, longer generations | 5 | ~124 |
| Ultra-conservative (lowest variance) | 3 | ~119 |

The container defaults to K=4 (best for tool-call agent workloads per FlashRT benchmarks).

---

## Docker Images Summary

| Image | Size | Contents |
|-------|------|----------|
| `flashrt:5090` | 29.7 GB | Base image: PyTorch 2.9 + CUDA 12.9 + FlashRT kernels (SM120) |
| `flashrt-server:5090` | 29.9 GB | Server image: base + fastapi + uvicorn + transformers + serving code |

---

## Known Limitations

| Limitation | Details |
|------------|---------|
| Greedy decode only | No temperature/top_p/top_k. Speculative decode verify uses argmax. The server accepts but ignores sampling params. |
| Tool call truncation | Fixed: `--default-max-tokens 8192` now baked into Dockerfile CMD. Previously defaulted to 2048, causing truncated outputs. Hard cap is `--max-output-tokens 32768`. |
| Single tool call per turn | Server stops generation on first complete tool call. Multiple tool calls in one response not captured. |
| NVFP4 hardcoded | Engine uses `quant="nvfp4"`. Switching to FP8 requires engine + kernel changes. |
| Thinking mode disabled | `enable_thinking` defaults to false. Enabling it may improve tool selection accuracy at cost of more tokens. |

---

## File Map

```
FlashRT/
├── Dockerfile.server          # Server image (healthcheck + CMD with --max-seq)
├── opencode.json              # opencode config (project-local)
├── qwen36_nvfp4/              # NVFP4 main model (~26 GB)
├── qwen36_fp8/                # FP8 checkpoint with mtp.safetensors (~29 GB)
├── serving/qwen36_agent/
│   ├── server.py              # FastAPI server (GET /v1/models/{id})
│   ├── service.py             # Core serving logic + tool call handling
│   │                          #   [local patch] cancel-on-disconnect queue + developer-role normalize
│   ├── engine.py              # AgentEngine protocol
│   │                          #   [local patch] generate_stream() cancel param
│   ├── qwen36_engine.py       # Qwen36 frontend adapter (greedy, nvfp4)
│   │                          #   [local patch] cancel-aware _committed_stream + generate_stream
│   ├── tool_stream.py         # Tool call XML/JSON parser
│   └── ...
├── docs/
│   ├── pi-developer-role.md   # Rationale for the developer-role patch
│   └── pi-developer-role.patch # Patch file (validate_messages only)
├── flash_rt/
│   ├── flash_rt_kernels.cpython-312-x86_64-linux-gnu.so
│   └── flash_rt_fa2.cpython-312-x86_64-linux-gnu.so
└── BUILD_NOTES.md             # This file
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Build hangs for hours | Ensure `JOBS=8` and `FA2_ARCH_NATIVE_ONLY=ON`. Uncapped parallelism OOMs in WSL2's 35 GB RAM |
| `ImportError: flash_rt_kernels` or `flash_rt_fa2` | Rebuild base image. Both `.so` files must land in `flash_rt/` |
| opencode says "Not Found" | Provider must be a **custom name** (e.g. `flashrt`), not the built-in `openai`. Must include `npm: "@ai-sdk/openai-compatible"` |
| Server unreachable from opencode | Container must bind `0.0.0.0` (not `127.0.0.1`). `baseURL` must include `/v1` suffix |
| `ModuleNotFoundError: serving.qwen36_agent.qwen36_engine` | Server image must COPY the full `serving/` directory and set `WORKDIR /workspace/FlashRT` |
| Slow tool calls | Model limitation at NVFP4 precision. Consider increasing `--default-max-tokens` via CLI (e.g., `--default-max-tokens 4096`) |
| **Client gets no response** | See [Client Disconnect / Zombie Process](#client-disconnect--zombie-process) below |
| **Server keeps running after client close** | Same root cause — the cancel-on-disconnect fix stops zombie decode within one speculative cycle |

---

## Client Disconnect / Zombie Process

### Symptoms

- You send a message via pi.dev or opencode. The server logs show it processing (stream in progress), but the client never shows a response.
- You close the client, but `docker logs` shows the generation continuing for minutes.
- Subsequent requests also appear stuck because the server is blocked on the zombie.

### Root cause

Two compounding issues:

1. **Throughput collapse at large context.** Decode speed drops from ~120 tok/s at short context to **3–5 tok/s** above 25K tokens. At 5 tok/s, a default 2048-token generation takes **~7 minutes**. Most clients timeout after 60–120 seconds.

2. **No cancellation on client disconnect.** Without the cancel-on-disconnect fix, the server's streaming path holds a single-thread lock (`service.py: self._lock`) for the entire generation. When the client disconnects, the GPU decode loop continues running until `max_tokens` is exhausted — a zombie. The lock blocks all subsequent requests until the zombie finishes. (This is the **unpatched** behavior; the fix in Step 2.1 uses a producer-consumer queue to avoid this.)

### The cascade

```
1. Request A starts → GPU decode at 5 tok/s
2. Client times out after 60s, disconnects
3. Server continues generating A as zombie (holds lock)
4. Request B arrives → blocks waiting for lock
5. Client B times out → B joins zombie queue
6. Server finishes zombie A → starts zombie B → ...
```

### Fix (applied 2026-06-07)

The cancel-on-disconnect mechanism spans three files:

**Architecture:** `stream_openai()` in `service.py` uses a **producer-consumer queue**
so SSE yields happen *outside* the GPU lock:

1. A daemon thread runs prefill + decode under `self._lock`, pushing SSE chunks into a `queue.Queue`.
2. The outer generator reads from the queue and yields SSE events — no lock held during I/O.
3. A `threading.Event` (`cancel`) is threaded through to `qwen36_engine.py`.

**Cancellation chain:**

1. On `GeneratorExit` (client disconnect), the outer generator's `except` block calls `cancel.set()`.
2. `_stream_openai()` runs `next(chunks)` in a fetcher thread with 100ms join-polling, so `cancel.is_set()` is checked even when the main thread is blocked on a CUDA kernel.
3. `_committed_stream()` in `qwen36_engine.py` checks `cancel.is_set()` between decode chunks and returns early.
4. `generate_stream()` checks `cancel.is_set()` between committed stream yields.
5. The `engine.py` protocol's `generate_stream()` signature includes the optional `cancel` parameter.

When detected, generation stops within **one speculative decode cycle** (~100ms) after
disconnect, and the lock is released. The zombie is bounded to ~1s at large context
instead of running for minutes.

### Evidence from logs

Before the fix, the container showed:

```
# TTFT of 80 seconds at 27K context — client already timed out
sid=frt-90799a0f38984543ab82ec1c | tok p=27271 | ttft=80234.8ms | decode=22349.8ms | speed=4.3 tok/s

# Decode of 118 seconds at 30K context
sid=frt-4eaf993ca359471a8a12f073 | tok p=30149 | decode=118257.6ms | speed=5.3 tok/s

# After this zombie, next request also blocked
```

### Mitigation beyond the cancel fix

If you still see slow responses at large contexts:

1. **Reduce context window.** Use `--max-seq 32768` instead of 262208 if your agent prompts stay under 32K.
2. **Reduce `--default-max-tokens`.** A lower budget (e.g., 4096) caps zombie duration even before the cancel fix kicks in.
3. **Start a new session.** Each client gets a new session by default; old sessions with large accumulated context won't slow new ones.
4. **Monitor with `docker logs -f`.** Watch the `tok/s` column. If it consistently drops below 5, the GPU is likely corrupt — the healthcheck should auto-restart.
5. **Use `--warmup-preset agent`** at startup to pre-capture common graph shapes.

### Recommended startup profiles

The 256K `--max-seq` in the Dockerfile is for the maximum possible context. For daily coding-agent
use, a smaller max-seq gives faster startup and lower memory pressure with no downside until the
conversation exceeds it:

```bash
# Fast coding agent (recommended for daily use)
# - 32K context covers ~25 tool-call turns before hitting the limit
# - TTFT stays under 3s for any prompt that fits
# - Decode: 78-98 tok/s
docker run --restart always --gpus all --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -p 8000:8000 -d \
    --name flashrt-qwen36 \
    -v $(pwd)/qwen36_nvfp4:/nvfp4:ro \
    -v $(pwd)/qwen36_fp8:/fp8:ro \
    -e FLASHRT_QWEN36_MTP_CKPT_DIR=/fp8 \
    -e FLASHRT_QWEN36_LONG_KV_CACHE=fp8 \
    -e FLASHRT_QWEN36_LONG_CTX_ROUTE_MIN_SEQ=512 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    flashrt-server:5090 \
    --max-seq 32768 --default-max-tokens 4096 --max-output-tokens 8192

# Long-context mode (for RAG / doc-QA / very long sessions)
# - 128K context, TTFT up to ~27s at full context
# - Decode: 70-78 tok/s at 100K+
docker run ... flashrt-server:5090 --max-seq 131072

# Maximum context (for testing 256K)
# - TTFT reaches 82s at 250K — expect client timeouts
# - Only useful with clients that have long timeout settings
docker run ... flashrt-server:5090 --max-seq 262208
```

### Don't use CUDA Graph replay for agent serving

The README benchmarks (144-175 tok/s) use `FLASHRT_QWEN36_TQ_VERIFY_GRAPH=1` and
`FLASHRT_QWEN36_TQ_MTP_CHAIN_GRAPH=1` with pre-warmed graph caches. This is a
fixed-shape benchmark mode. In a live agent session:

- Every decode step is a new position → unique graph key → cold capture cost
- Graph state accumulates and can corrupt, causing `cudaErrorIllegalAddress`
- We observed this corruption: after graph-replay experiments, subsequent requests
  ran at 3-5 tok/s until container restart

The production default (graph flags off, direct kernel launch) gives **78-98 tok/s**
reliably at typical agent contexts, with no corruption risk.

---

## GPU Health Monitoring

The Dockerfile includes a decode-speed-aware healthcheck that goes beyond simple
liveness checks. It runs every 20 seconds (after a 90-second start-period) and:

1. Sends a tiny `/v1/chat/completions` request (`"ping"`, `max_tokens=5`).
2. Reads `flashrt.decode_tok_per_s` from the response.
3. If throughput is **> 0 but < 5 tok/s** (GPU severely degraded but still responding),
   kills PID 1 via `SIGTERM` → Docker's `--restart always` restarts the container.
4. If the request itself fails (CUDA error, timeout, crash), also kills PID 1.

This catches the "GPU corrupt" scenario where the server is technically alive but
producing at 3-5 tok/s (e.g., after CUDA graph state corruption). Without it,
Docker considers the container healthy because HTTP 200 is still returned.

The threshold is intentionally conservative at 5 tok/s. During testing, we observed
that after a large-context session (50K+ tokens), tiny 5-token healthcheck outputs
regularly dip to 10-20 tok/s due to residual KV cache memory pressure. Setting the
threshold any higher (e.g., 10 or 20) causes **false-positive restarts** that kill
a healthy server and disconnect the client mid-session.

| Healthcheck parameter | Value | Why |
|-----------------------|-------|-----|
| `--interval` | 60s | Avoids false positives and reduces GPU cycles wasted on monitoring |
| `--timeout` | 15s | Accounts for slow decode at large context |
| `--start-period` | 90s | Model load + MTP head takes ~30s, plus first warmup |
| `--retries` | 1 | Kill immediately on first failure — no point retrying a corrupt GPU |
