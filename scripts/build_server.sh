#!/usr/bin/env bash
# Build flashrt-server:5090 from Dockerfile.server, then retag it as
# flashrt:5090 so the FROM line keeps working for the next build.
#
# Usage:
#   ./scripts/build_server.sh                     # standard build
#   ./scripts/build_server.sh --no-cache          # force full rebuild
#   ./scripts/build_server.sh --no-retag          # build only
#
# What it does:
#   1. Reads the CMD line from Dockerfile.server
#   2. Runs docker build (or shows what would change)
#   3. Retags the new image as flashrt:5090 (so the FROM line resolves)
#   4. Reports the final image IDs and CMD

set -euo pipefail

NO_CACHE=""
NO_RETAG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-cache) NO_CACHE="--no-cache"; shift ;;
        --no-retag) NO_RETAG="1"; shift ;;
        -h|--help)   sed -n '2,16p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1"; exit 1 ;;
    esac
done

# Extract the CMD from Dockerfile.server so we can verify the build picks it up
EXPECTED_CMD=$(grep -E '^CMD ' Dockerfile.server | head -1)
echo "==> Dockerfile.server CMD line:"
echo "    $EXPECTED_CMD"

# Capture old image IDs (if any) so we can show the diff
OLD_SERVER_ID=$(docker inspect --format='{{.Id}}' flashrt-server:5090 2>/dev/null || echo "")
OLD_BASE_ID=$(docker inspect --format='{{.Id}}' flashrt:5090 2>/dev/null || echo "")

echo ""
echo "==> Building flashrt-server:5090 (this is fast — ~1s if no source changed) ..."
docker build $NO_CACHE -t flashrt-server:5090 -f Dockerfile.server .

NEW_SERVER_ID=$(docker inspect --format='{{.Id}}' flashrt-server:5090)
echo ""
echo "==> Build complete"
echo "    flashrt-server:5090 = $NEW_SERVER_ID"
if [[ -n "$OLD_SERVER_ID" && "$OLD_SERVER_ID" != "$NEW_SERVER_ID" ]]; then
    echo "    (was $OLD_SERVER_ID)"
else
    echo "    (no change in image ID — only metadata updated)"
fi

if [[ -n "$NO_RETAG" ]]; then
    echo ""
    echo "==> --no-retag set, skipping the base retag. The next build's FROM will use"
    echo "    the OLD base ($OLD_BASE_ID) until you retag."
    exit 0
fi

# Retag the new image as flashrt:5090 so the FROM line keeps working
echo ""
echo "==> Retagging flashrt-server:5090 → flashrt:5090 (stable base for next build) ..."
docker tag flashrt-server:5090 flashrt:5090
NEW_BASE_ID=$(docker inspect --format='{{.Id}}' flashrt:5090)
echo "    flashrt:5090          = $NEW_BASE_ID"

echo ""
echo "==> CMDs:"
docker inspect --format='{{json .Config.Cmd}}' flashrt-server:5090 | python3 -c "import json,sys; print('    flashrt-server:5090  ', json.load(sys.stdin))"
docker inspect --format='{{json .Config.Cmd}}' flashrt:5090          | python3 -c "import json,sys; print('    flashrt:5090         ', json.load(sys.stdin))"

echo ""
echo "==> Disk usage:"
docker system df | head -3

echo ""
echo "==> Next: docker run -d --name flashrt-qwen36 --gpus all --network=host \\"
echo "         --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \\"
echo "         -v /home/tzuto/Projects/FlashRT/qwen36_nvfp4:/nvfp4:ro \\"
echo "         -v /home/tzuto/Projects/FlashRT/qwen36_fp8:/fp8:ro \\"
echo "         -e FLASHRT_QWEN36_MTP_CKPT_DIR=/fp8 \\"
echo "         -e FLASHRT_QWEN36_LONG_KV_CACHE=fp8 \\"
echo "         -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\"
echo "         -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \\"
echo "         flashrt-server:5090"
