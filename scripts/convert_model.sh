#!/usr/bin/env bash
# Download Huihui-gemma-4-E2B Q4_K (~3.4 GB) directly from HuggingFace.
# No Python, no PyTorch, no conversion needed — wget only.
# Supports resume (-c) if download is interrupted.
#
# Usage: ./scripts/download_model.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_DIR/models"

HF_URL="https://huggingface.co/huihui-ai/Huihui-gemma-4-E2B-it-qat-q4_0-unquantized-abliterated-GGUF/resolve/main/Huihui-gemma-4-E2B-it-qat-q4_0-unquantized-abliterated-Q4_K.gguf"
OUT_FILE="$MODELS_DIR/model.gguf"

mkdir -p "$MODELS_DIR"

if [[ -f "$OUT_FILE" ]]; then
    echo "==> $OUT_FILE already exists, skipping download."
    echo "    Delete it manually to re-download."
    exit 0
fi

echo "==> Downloading Huihui-gemma-4-E2B Q4_K (~3.4 GB)..."
echo "    (download can be interrupted and resumed)"
wget -c "$HF_URL" -O "$OUT_FILE" --show-progress

echo ""
echo "==> Done: $OUT_FILE"
echo "    Run: cd \"$PROJECT_DIR\" && ./build/ai-agent"
