#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

VIZ="${1:-}"
ARGS=(-c configs/demos/smbu_seq2.yaml)
[[ "${VIZ:-}" == "--viz" ]] && ARGS+=(--viz)
python main.py "${ARGS[@]}"
