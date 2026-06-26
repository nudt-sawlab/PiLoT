#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

NAME="${1:-3dgs_test}"
VIZ="${2:-}"
ARGS=(-c configs/demos/feicuiwan.yaml --name "$NAME")
[[ "${VIZ:-}" == "--viz" ]] && ARGS+=(--viz)
python main.py "${ARGS[@]}"
