# Pi (pi-coding-agent) compatibility: `developer` role support

## Problem

The [pi coding agent](https://pi.dev) sends OpenAI-style requests that include a `developer` role message (used for reasoning-style system instructions). FlashRT's `/v1/chat/completions` endpoint initially rejected the `developer` role outright, and even after allowing it, Qwen's chat template rejected requests that produced two `system` messages (e.g. when a request had both an explicit `system` and a `developer` that got naively converted to `system`).

This caused 400 / 500 errors with messages like:

- `unsupported role: 'developer'`
- `jinja2.exceptions.TemplateError: System message must be at the beginning.`

## Fix

`validate_messages` in `serving/qwen36_agent/service.py` now normalizes incoming messages by:

1. **Merging `system` + `developer` messages** into a single leading `system` message.
   Some chat templates (Qwen's, Anthropic's) reject multiple system messages or a non-leading system message. Pi's openai-completions transport sends a `developer` role for OpenAI-style reasoning prompts, which we fold into the leading system message.

2. **Flattening list-style content blocks** (`[{"type": "text", "text": "..."}]`) into a plain string.
   OpenAI accepts both string and list content, but pi always sends list-style content blocks (so it can interleave text and images). The Qwen chat template only handles string content.

## Apply the patch

From the repo root:

```bash
git apply docs/pi-developer-role.patch
```

## Why it was needed for pi specifically

In pi's `~/.pi/agent/models.json`, the FlashRT provider has `compat.supportsDeveloperRole: true`. This tells pi it's safe to send `developer` messages. The agent also sends a `system` message in many cases (e.g. with `--system-prompt` or with tool schemas). The merge logic in this patch handles both cases.

## Testing

The patch was validated end-to-end against the running Docker container (`flashrt-qwen36`) with:

- A plain `developer` + `user` request
- A `system` + `developer` + `user` request
- A `system` + `developer` + `user` request with `tools` (a function-calling turn)
- A request with list-style `content` blocks (`[{"type":"text", "text":"..."}]`) - this is the actual format pi sends
- Running `pi -p` end-to-end against `qwen36-27b` via FlashRT

All cases return successful completions.

## Workflow: Option A (committed) vs Option B (patch file)

This repo currently uses **Option B**: the patch is stored as
`docs/pi-developer-role.patch` and applied at build time. The patched source
file (`serving/qwen36_agent/service.py`) is left unmodified in git.

### Why Option B was chosen

- This fork tracks upstream `LiangSu8899/FlashRT` via periodic `git pull`.
- Keeping the patch in the working tree (not in a commit) means pulls never
  conflict with our local change.
- The patch is small (~23 lines) and re-applying it after a pull is one line:
  `git apply docs/pi-developer-role.patch`.

### When to switch to Option A (commit the change)

If you stop tracking upstream and maintain your own fork, you can switch to a
cleaner setup:

1. `git checkout -- serving/qwen36_agent/service.py` (revert the local changes)
2. `git apply docs/pi-developer-role.patch` (re-apply, just to be safe)
3. `git add serving/qwen36_agent/service.py`
4. Commit and remove `docs/pi-developer-role.patch` from future builds
5. Update `BUILD_NOTES.md` Step 2.1 to drop the `git apply` step

