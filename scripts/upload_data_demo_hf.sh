#!/usr/bin/env bash
# Upload data_demo/ to Hugging Face dataset choyaa/PiLoT-data (resumable chunks).
# Prerequisite: hf auth login
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="choyaa/PiLoT-data"
DD="$ROOT/data_demo"
CITYGS_SMBU="${CITYGAUSSIAN_SMBU:-/home/choya/Documents/code/CityGaussian/3dgs_model/SMBU}"
MODE="${1:-remaining}"

if ! hf auth whoami &>/dev/null; then
  echo "Not logged in. Run: hf auth login"
  exit 1
fi

upload() {
  local src="$1" dst="$2"
  if [[ ! -e "$src" ]]; then
    echo "SKIP (missing): $src"
    return 0
  fi
  echo ">>> $src -> $dst"
  hf upload "$REPO" "$src" "$dst" --repo-type dataset
}

case "$MODE" in
  all)
    upload "$DD/3dgs_model" "data_demo/3dgs_model"
    upload "$DD/pretrained_model" "data_demo/pretrained_model"
    upload "$DD/query/poses" "data_demo/query/poses"
    upload "$DD/query/images/3dgs_test" "data_demo/query/images/3dgs_test"
    upload "$DD/query/images/smbu_seq2" "data_demo/query/images/smbu_seq2"
    upload "$CITYGS_SMBU/checkpoints" "data_demo/smbu_model/checkpoints"
    upload "$CITYGS_SMBU/sparse" "data_demo/smbu_model/sparse"
    upload "$DD/README.md" "data_demo/README.md"
    ;;
  remaining)
    # Skip parts already on HF (3dgs_model, pretrained_model uploaded 2026-06-26)
    upload "$DD/query/poses" "data_demo/query/poses"
    upload "$DD/query/images/3dgs_test" "data_demo/query/images/3dgs_test"
    upload "$DD/query/images/smbu_seq2" "data_demo/query/images/smbu_seq2"
    upload "$CITYGS_SMBU/checkpoints" "data_demo/smbu_model/checkpoints"
    upload "$CITYGS_SMBU/sparse" "data_demo/smbu_model/sparse"
    upload "$DD/README.md" "data_demo/README.md"
    ;;
  *)
    echo "Usage: $0 [all|remaining]"
    exit 1
    ;;
esac

echo "Done. https://huggingface.co/datasets/$REPO"
