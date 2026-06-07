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

```bash
# 1. Stash any local working-tree changes (the pi patch)
git stash push -m 'local patches' -- serving/qwen36_agent/service.py

# 2. Fast-forward to upstream
git fetch upstream
git merge --ff-only upstream/main

# 3. Re-apply the pi patch (see Step 2.1 for full details)
git apply docs/pi-developer-role.patch

# 4. Push the upstream commits to your fork (keeps fork in sync)
git push origin main
```

The pi patch is kept in your **working tree only** (not committed) so pulls
never conflict with your local change. See Step 2.1 for the full re-apply
workflow including conflict resolution.

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
docker build -t flashrt:5090 \
    -f docker/Dockerfile \
    --build-arg GPU_ARCH=120 \
    --build-arg JOBS=8 \
    --build-arg FA2_ARCH_NATIVE_ONLY=ON \
    .
```

### Build Notes

| Setting | Value | Why |
|---------|-------|-----|
| `JOBS=8` | 8 parallel nvcc threads | Peak ~16-24 GB RAM during compilation. With 35 GB WSL2 RAM, uncapped parallelism (default = nproc) causes OOM or multi-hour hangs |
| `FA2_ARCH_NATIVE_ONLY=ON` | Skip sm_80 + compute120 PTX | Only builds native SM120 SASS for FlashAttention2. ~66% less FA2 compilation work |
| `GPU_ARCH=120` | Target RTX 5090 | Required for Blackwell support |

Expected build time: **~5 minutes** for CUDA kernel compilation (282s measured).

Smoke test inside the image:
```bash
docker run --rm --gpus all flashrt:5090 \
    python3 -c "import flash_rt; print('flash_rt', flash_rt.__version__); from flash_rt import flash_rt_kernels; print('kernels OK')"
```

Expected output: `flash_rt 0.1.0` + `kernels OK`

---

## Step 2: Build the Server Image

Extends the base image with FastAPI, uvicorn, and the serving code baked in.
No pip install at runtime — faster startup.

```bash
# From the FlashRT repo root
docker build -t flashrt-server:5090 -f Dockerfile.server .
```

`Dockerfile.server`:
```dockerfile
FROM flashrt:5090

RUN pip install --quiet --no-cache-dir fastapi uvicorn 'transformers<4.56'

COPY serving/ /workspace/FlashRT/serving/
WORKDIR /workspace/FlashRT

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=60s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/v1/models')" || exit 1

CMD ["python3", "-m", "serving.qwen36_agent.server", "--checkpoint", "/nvfp4", "--max-seq", "262208", "--host", "0.0.0.0", "--port", "8000"]
```


### Step 2.1: Apply custom patches

Some external clients (e.g. the [pi coding agent](https://pi.dev)) send OpenAI-style
`developer` role messages and list-style `content` blocks that FlashRT's Qwen chat
template rejects. Apply the following patch before building the server image:

```bash
git apply docs/pi-developer-role.patch
docker build -t flashrt-server:5090 -f Dockerfile.server .
```

Without this patch, requests from pi will fail with 400 errors and messages like:
`unsupported role: 'developer'`, `jinja2.exceptions.TemplateError: System message
must be at the beginning.`, or `message.content must be a string`.

See `docs/pi-developer-role.md` for the full rationale.

#### Re-applying the patch after a `git pull`

The patch is kept in `docs/pi-developer-role.patch` (a text file in the repo)
rather than committed as inline code, so you can `git pull upstream` without
merge conflicts. The patched file is in your working tree only. To re-apply
after pulling upstream:

```bash
git pull                          # fetch upstream
git apply docs/pi-developer-role.patch   # may say "patch does not apply"
```

If `git apply` reports the patch is already applied (e.g. you forgot to revert
before pulling, or the upstream already includes the fix), first revert then
re-apply cleanly:

```bash
git checkout -- serving/qwen36_agent/service.py
git apply docs/pi-developer-role.patch
```

If upstream changed lines that the patch touches, `git apply` may fail with
rejects. In that case, open the patched region in an editor and re-apply the
intent of the patch manually (the patch is small, ~23 lines).

**Alternative workflow (commit the patch, drop the patch file):** if you'd rather
have the patched code committed and you don't sync with upstream often, see
`docs/pi-developer-role.md` for a note on the trade-off.

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
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    flashrt-server:5090
```

| Flag | Purpose |
|------|---------|
| `--restart always` | Auto-starts with Docker Desktop, auto-restarts on crash |
| `--gpus all` | GPU passthrough |
| `--ipc=host` | Shared memory for CUDA |
| `-p 8000:8000` | Expose server port |
| `MTP_CKPT_DIR=/fp8` | MTP head location (speculative decode) |
| `LONG_KV_CACHE=fp8` | FP8 KV cache for 256K context |
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
| Tool call truncation | Default `--default-max-tokens` is 2048. Tool calls truncated before closing tags become plain text. Consider increasing to 4096+. |
| Single tool call per turn | Server stops generation on first complete tool call. Multiple tool calls in one response not captured. |
| NVFP4 hardcoded | Engine uses `quant="nvfp4"`. Switching to FP8 requires engine + kernel changes. |
| Thinking mode disabled | `enable_thinking` defaults to false. Enabling it may improve tool selection accuracy at cost of more tokens. |

---

## File Map

```
FlashRT/
├── Dockerfile.server          # Server image definition
├── opencode.json              # opencode config (project-local)
├── qwen36_nvfp4/              # NVFP4 main model (~26 GB)
├── qwen36_fp8/                # FP8 checkpoint with mtp.safetensors (~29 GB)
├── serving/qwen36_agent/
│   ├── server.py              # FastAPI server (patched: added GET /v1/models/{id})
│   ├── service.py             # Core serving logic + tool call handling
│   ├── tool_stream.py         # Tool call XML/JSON parser
│   ├── qwen36_engine.py       # Qwen36 frontend adapter (greedy, nvfp4)
│   └── ...
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
| `ImportError: flash_rt_kernels` | Rebuild base image. `.so` files must land in `flash_rt/` |
| opencode says "Not Found" | Provider must be a **custom name** (e.g. `flashrt`), not the built-in `openai`. Must include `npm: "@ai-sdk/openai-compatible"` |
| Server unreachable from opencode | Container must bind `0.0.0.0` (not `127.0.0.1`). `baseURL` must include `/v1` suffix |
| `ModuleNotFoundError: serving.qwen36_agent.qwen36_engine` | Server image must COPY the full `serving/` directory and set `WORKDIR /workspace/FlashRT` |
| Slow tool calls | Model limitation at NVFP4 precision. Consider increasing `--default-max-tokens` from 2048 to 4096 |
