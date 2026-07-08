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
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8765/health', timeout=5)" || exit 1

CMD ["python3", "-m", "serving.qwen36_agent.server", "--checkpoint", "/nvfp4", \
     "--max-seq", "196608", "--route-min-seq", "0", \
     "--default-max-tokens", "32768", "--max-output-tokens", "65536", \
     "--host", "0.0.0.0", "--port", "8000"]
```

### Step 2.1: Local patches (committed to this fork)

| Change | Files | Purpose |
|--------|-------|---------|
| Developer-role normalize | `service.py` | Normalize `developer` → `system`, flatten list-style content blocks |
| Cancel-on-disconnect | `service.py`, `qwen36_engine.py`, `engine.py` | Stop zombie GPU decode after client disconnect |
| Deterministic GPU release on `docker stop` | `service.py`, `qwen36_engine.py`, `server.py` | Cancel in-flight streams on SIGTERM and tear the CUDA context down so VRAM is returned to the driver every time (see [GPU Memory Not Released on `docker stop`](#gpu-memory-not-released-on-docker-stop) below) |

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

**Manual start, no auto-restart, port 8765.** WSL2 + Docker Desktop's port
forwarder (`/forwards/expose`) returns HTTP 500 for port 8000 specifically —
use 8765 on the host side (container still listens on 8000 internally).
No `--restart` flag: the container only runs when you explicitly start it.

```bash
docker run --gpus all --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --stop-timeout 30 \
    -p 8765:8000 -d \
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

| Flag | Value | Why |
|------|-------|-----|
| `-p 8765:8000` | Bridge port mapping | Docker Desktop's WSL2 forwarder returns 500 for host port 8000; 8765 works |
| _(no `--restart`)_ | `no` (default) | Manual start only — container does not survive crashes or reboots automatically |
| `--gpus all` | GPU passthrough | Required for CUDA on Blackwell |
| `--ipc=host` | Shared IPC | CUDA IPC for efficient GPU memory sharing |

The Dockerfile bakes in:
- `--max-seq 196608` (192K context window)
- `--route-min-seq 0` (FP8-KV path for all prompts — avoids per-position graph capture)
- `--default-max-tokens 32768` (default output budget)
- `--max-output-tokens 65536` (hard cap; input + output share the 192K pool dynamically)

| Env var | Purpose |
|---------|---------|
| `MTP_CKPT_DIR=/fp8` | MTP head location (speculative decode) |
| `LONG_KV_CACHE=fp8` | FP8 KV cache for long context |
| `HF_HUB_OFFLINE=1` | Don't phone home to HuggingFace |

`--stop-timeout 30` gives the SIGTERM-driven GPU teardown (see
[GPU Memory Not Released on `docker stop`](#gpu-memory-not-released-on-docker-stop))
ample margin over Docker's default 10 s grace. The teardown itself is bounded to
~one decode cycle (active streams are cancelled first), so `docker stop`
normally completes in 1–3 s.

Startup takes ~10s. Verify:

```bash
curl -s http://localhost:8765/v1/models/qwen36-27b
# {"id":"qwen36-27b","object":"model","owned_by":"flash-rt","context_length":196608,"max_output_tokens":65536}

curl -s http://localhost:8765/v1/chat/completions \
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

## Step 5: Configure clients (pi / opencode / hermes)

The server reports `context_length=196608` and `max_output_tokens=65536` via
`/v1/models`, but most clients **do not auto-query the server** — they read a
static `contextWindow`/`maxTokens` from their own config at startup. So every
client config must be set to match the server, or the client's meter / compaction
will use the wrong numbers (and an over-large client `maxTokens` won't actually
raise the budget the server honors for *other* clients).

| value | server | what clients must set |
|---|---|---|
| context window | `--max-seq 196608` | `contextWindow` / `context_length` / `limit.context` = **196608** |
| output budget | `--default-max-tokens 32768` (cap `--max-output-tokens 65536`) | `maxTokens` / `limit.output` = **32768** (see note below) |

### opencode

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
        "baseURL": "http://127.0.0.1:8765/v1",
        "apiKey": "-",
        "headers": { "Authorization": "Bearer -" }
      },
      "models": {
        "qwen36-27b": {
          "name": "Qwen3.6 27B NVFP4 (local RTX 5090)",
          "tool_call": true,
          "temperature": false,
          "limit": { "context": 196608, "output": 32768 },
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
| `baseURL` | `http://127.0.0.1:8765/v1` | Must include `/v1`. Server binds `0.0.0.0` for WSL2 |
| `limit.context` / `limit.output` | `196608` / `32768` | Must match the server (see output-budget note) |

### pi

`~/.pi/agent/models.json` (WSL2) and `C:\Users\<you>\.pi\agent\models.json`
(Windows) — register the `flashrt` provider with the same numbers:

```json
{
  "providers": {
    "flashrt": {
      "baseUrl": "http://127.0.0.1:8765/v1",
      "api": "openai-completions",
      "apiKey": "-",
      "compat": { "supportsDeveloperRole": true },
      "models": [{
        "id": "qwen36-27b",
        "name": "Qwen3.6 27B NVFP4 (local RTX 5090)",
        "contextWindow": 196608,
        "maxTokens": 32768,
        "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 }
      }]
    }
  }
}
```

Restart pi after editing — it reads `models.json` at launch, not per request.

### hermes

`~/.hermes/config.yaml` (WSL2) and `C:\Users\<you>\AppData\Local\hermes\config.yaml`
(Windows). Two non-obvious rules:

1. **Per-model value, not global.** Put `context_length` under
   `custom_providers[].models.<model>`, **not** at the top-level `model:` block.
   A top-level `model.context_length` *overrides* the per-provider value (hermes
   checks the global first), so a leftover global `context_length: 1000000` will
   make the status bar show 1M for every provider. Omit the global override.
2. **No `max_tokens` on the flashrt model.** hermes reads `max_tokens` only from
   the global `model.max_tokens` / `HERMES_MAX_TOKENS` env (no per-provider
   field), and setting it globally would also cap other providers (e.g. glm).
   Leave it unset — hermes then omits `max_tokens` in requests and inherits the
   **server default (32768)**, which is exactly what you want.

```yaml
custom_providers:
- name: flashrt                       # meaningful name (NOT a UUID)
  base_url: http://127.0.0.1:8765/v1
  api_key: '-'
  models:
    qwen36-27b:
      name: Qwen3.6 27B NVFP4 (local RTX 5090)
      context_length: 196608          # MUST be under models.<model>, not top-level
  model: qwen36-27b
```

### cc-switch (if you manage providers via cc-switch)

cc-switch writes its `providers` table into the agent configs. The provider's
**`id`** becomes the name written into the hermes config — so a UUID `id` leaks
into `~/.hermes/config.yaml` as `provider: 7b2ca8fd-…`. Use a meaningful `id`
(`flashrt`) so all configs stay consistent. The `providers` PK is
`(id, app_type)`, so `flashrt`/hermes does not collide with `flashrt`/opencode.
The opencode-provider row stores the model limit in `settings_config`
(`models.qwen36-27b.limit.{context,output}`) — keep it at 196608 / 32768.

### Output-budget note (why 32768, and the cap-vs-default distinction)

The server has **two** output knobs:

- `--default-max-tokens 32768` — used **only** when a request omits `max_tokens`
  (this is what hermes does). Raising this affects hermes and any client that
  omits the field.
- `--max-output-tokens 65536` — the **hard ceiling**. Any request above this is
  clamped down (not rejected).

Clients that always send `max_tokens` (pi, opencode) honor their own value up to
the hard cap, so raising the server *default* alone does **not** affect them —
you must set the client's `maxTokens`/`limit.output`. That's why both the server
default and every client are set to **32768** for consistency: a generous
budget for long summaries, still well under the 65536 ceiling. If you only ever
use pi, you can lower the server default back to 8192 (the client value governs
pi) without affecting it.

---

## Performance

All numbers measured on RTX 5090 (SM120, 32 GB VRAM) in WSL2 Docker,
`--max-seq 196608`, `--route-min-seq 0`, FP8-KV, speculative decode K=4,
CUDA graph flags off (production default).

| Metric | Value |
|--------|-------|
| Cold startup | ~10s |
| Short-prompt decode | **~110 tok/s** |
| 64K context decode | **~89 tok/s** |
| 200K context decode | **~97 tok/s** |
| Without speculative decode | ~36 tok/s (K=1, no MTP) |
| Context window | 192K tokens |
| Default output budget | 32,768 (server default; matches pi/hermes configs) |
| Max output tokens (hard cap) | 65,536 |
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

### Why 196608 (192K) instead of 128K / 224K / 256K?

This is a deliberate trade between **context capacity** and **decode-speed
stability** on a 32 GB card shared with the Windows desktop (WSL2). Real
pi/coding-agent sessions that paste whole articles or large files routinely
grow past 95K tokens; at `--max-seq 131072` (128K) those sessions overflowed
`max_seq` mid-request and the server raised `ValueError` mid-SSE, surfacing to
pi as a broken `Stream ended without finish_reason`.

`--max-seq 196608` (192K) comfortably fits a 95K session + multi-turn growth +
the 32768 default output budget, with measured steady-state decode of
~108–120 tok/s up to 32K real context. It leaves ~1.3 GB idle free VRAM —
borderline, so under extreme session growth + an unlucky WSL2 free-VRAM drift
you may see decode slow to ~30 tok/s; a `docker restart` clears it.

The 240K cliff we identified in `qwen36_maxseq_sweep.py`:
```
max_seq   free_MiB   decode tok/s (1K ctx)
196608      844     122
229376      944     119   <- last safe
245760      364      27   <- cliff
```

We briefly tried `--max-seq 229376` (224K) thinking it was a free upgrade
(see benchmarks/qwen36_maxseq_results.md for the full data). It turned out
to be equivalent to 192K at every input context (within 0-6%) but 22K from
the cliff. The Phase A "free 32K capacity" was unused — no production
workload uses 200K-224K context. Reverted to 192K for the larger cliff
headroom (60K vs 22K).

**Why not 256K?** The 256K persistent cache is 5.4 GB vs 4.6 GB at 224K,
enough to cross the cliff. Even with the tighter-stage env vars
(`FLASHRT_QWEN36_FP8_STAGE_CAP=131072`, etc.) decode collapses to 7-15 tok/s
at 64K+ input context — 10× slower than 192K. To get past 240K requires
compressing the persistent cache itself (TQ bit-pack, future option), not
trimming the staging buffers.

Two guards make 192K safe to run as the default:

1. **Overflow no longer crashes the stream.** `service._reclip_max_tokens_for_engine`
   re-clips the output budget against the ACTUAL engine token count (the hot
   session journal, which can be longer than the client-rendered prompt because
   of hidden Qwen3.6 control tokens). If a session genuinely fills `max_seq`,
   the server returns a clean HTTP 400 ("the active conversation is N tokens
   long, which already fills --max-seq; start a new session") instead of a
   broken SSE stream.
2. **`docker stop` releases VRAM deterministically** (see
   [GPU Memory Not Released on `docker stop`](#gpu-memory-not-released-on-docker-stop)),
   so recovery from the slow state is a clean ~10 s stop/start, not a
   `wsl --shutdown`.

If your sessions are small (rarely pasting whole files) and you want rock-solid
decode speed with no restarts, `--max-seq 131072` (128K, ~3.5 GB free) is the
bulletproof choice. The earlier sweep data is in
`benchmarks/qwen36_graph_results.md` and the new 192K-vs-224K-vs-256K
comparison is in `benchmarks/qwen36_maxseq_results.md`.

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
| VRAM not freed after `docker stop` | See [GPU Memory Not Released on `docker stop`](#gpu-memory-not-released-on-docker-stop) below |

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

## GPU Memory Not Released on `docker stop`

### The problem

When a container is stopped, Docker sends `SIGTERM`, waits its grace period
(10 s default), then `SIGKILL`. FlashRT loads ~31 GB of weights + KV cache as
CUDA tensors, and a streaming request can generate up to 65 536 tokens (tens of
seconds). Plain `uvicorn.run()` does *graceful* shutdown: it **waits for
in-flight requests to finish** before running the ASGI lifespan shutdown. So a
live generation kept uvicorn alive past the 10 s grace, Docker `SIGKILL`ed the
process **mid-CUDA-operation**, and on WSL2 / Docker Desktop the `dxgkrnl`
GPU paravirtualization layer intermittently failed to reclaim the VRAM — the
"dedicated GPU memory not released" you saw, only fixed by restarting the WSL
VM or Docker Desktop.

There was also **no explicit CUDA teardown anywhere**: weights, CUDA graphs,
the graph mempool, and streams were all left for the OS to reclaim.

### The fix

Deterministic, fast shutdown so the process always exits cleanly (never
`SIGKILL`ed mid-op) and the CUDA context is torn down explicitly:

1. **Cancel in-flight streams on SIGTERM, before uvicorn drains connections.**
   `main()` runs uvicorn through a `Server` subclass (`_GracefulGpuServer`)
   whose `SIGTERM`/`SIGINT` handler calls `service.request_shutdown()` — a
   non-blocking (signal-safe) flip of a `threading.Event` plus one `.set()`
   per active stream. Streaming consumers poll this every 0.2 s, so every
   response ends within ~one decode cycle and uvicorn shuts down well inside
   Docker's grace.
2. **Join the daemon decode threads.** `service.shutdown()` waits (bounded,
   ~3 s) for the producer threads to observe `cancel` and exit, so no decode
   kernel is in flight when the context is torn down.
3. **Explicit CUDA teardown.** `qwen36_engine.release_gpu()` calls
   `torch.cuda.synchronize()`, clears the graph caches + mempool, drops the
   frontend (weights / streams / KV cache / HF pipeline), runs `gc.collect()`
   + `torch.cuda.ipc_collect()`, then `torch.cuda.empty_cache()` — handing all
   device memory back to the driver. This runs in the FastAPI **lifespan
   shutdown** hook, and again in `main()`'s `finally` (covers `force_exit`).

Net effect: `docker stop` completes in ~1–3 s every time and VRAM is returned
on the spot. Both `shutdown()` and `release_gpu()` are idempotent.

### Verify

```bash
# Watch VRAM before / during / after stop (Windows host, or nvidia-smi in WSL)
nvidia-smi --query-gpu=memory.used --format=csv -l 1
docker stop flashrt-qwen36    # ~1–3 s, then memory drops back to ~0
```

You should see the `release_gpu: CUDA context torn down` line in
`docker logs flashrt-qwen36` right before the container exits.

### Operational notes

- `--stop-timeout 30` in `docker run` (shown in [Step 4](#step-4-start-the-server-container))
  gives margin; the teardown is bounded to ~one decode cycle so it rarely
  needs it.
- A second `SIGTERM`/Ctrl-C sets uvicorn `force_exit`, which skips the
  lifespan hook — the `main()` `finally` block still runs the teardown.
- If VRAM ever still appears held after stop, `docker ps -a` for a zombie
  container or check `nvidia-smi`'s process list; a `wsl --shutdown` clears
  any leftover WSL2 GPU state.

---

## GPU Health Monitoring

The healthcheck uses `GET /health` — returns `{"status":"ok"}` without GPU
inference. Zero VRAM overhead, zero interference with active requests. The
container is **manually started** (no `--restart` policy); if the server
crashes (CUDA error, OOM, segfault), you'll see it in `docker logs` and
restart by hand with `docker start flashrt-qwen36`.

| Parameter | Value | Why |
|-----------|-------|-----|
| `--interval` | 60s | Standard cadence |
| `--timeout` | 10s | `/health` is instant |
| `--start-period` | 90s | Model load + margin |
| `--retries` | 2 | One transient failure tolerated |
