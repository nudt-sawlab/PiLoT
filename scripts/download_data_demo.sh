#!/usr/bin/env bash
# Download data_demo/ from Hugging Face into the repository root.
#
#   ./scripts/download_data_demo.sh          # full demo data (~15GB)
#   ./scripts/download_data_demo.sh feicuiwan  # Feicuiwan only (~4.5GB)
#
# Requires: pip install huggingface_hub  (hf download)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
REPO="choyaa/PiLoT-data"
MODE="${1:-all}"

if ! command -v hf &>/dev/null; then
  echo "Install Hugging Face CLI: pip install huggingface_hub"
  exit 1
fi

case "$MODE" in
  all|"")
    PATTERNS=("data_demo/**")
    ;;
  feicuiwan)
    PATTERNS=(
      "data_demo/3dgs_model/**"
      "data_demo/pretrained_model/**"
      "data_demo/query/images/3dgs_test/**"
      "data_demo/query/poses/3dgs_test.txt"
      "data_demo/README.md"
    )
    ;;
  smbu)
    PATTERNS=(
      "data_demo/smbu_model/**"
      "data_demo/pretrained_model/**"
      "data_demo/query/images/smbu_seq2/**"
      "data_demo/query/poses/smbu_seq2.txt"
      "data_demo/README.md"
    )
    ;;
  *)
    echo "Usage: $0 [all|feicuiwan|smbu]"
    exit 1
    ;;
esac

echo "Downloading from https://huggingface.co/datasets/$REPO"
echo "Target: $ROOT/data_demo/"
echo "Mode: $MODE"

ARGS=(download "$REPO" --repo-type dataset --local-dir "$ROOT")
for p in "${PATTERNS[@]}"; do
  ARGS+=(--include "$p")
done

hf "${ARGS[@]}"

echo ""
echo "Download complete."
echo "  Feicuiwan: data_demo/3dgs_model/ + query/images/3dgs_test/"
echo "  SMBU:      data_demo/smbu_model/ + query/images/smbu_seq2/"
echo "  Shared:    data_demo/pretrained_model/"
