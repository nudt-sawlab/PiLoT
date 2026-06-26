#!/usr/bin/env bash
# Shared setup for all PiLoT demo scripts.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -z "${CITYGAUSSIAN_ROOT:-}" ]]; then
  if [[ -d "$ROOT/third_party/CityGaussian" ]]; then
    CITYGAUSSIAN_ROOT="$ROOT/third_party/CityGaussian"
  elif [[ -d "$ROOT/../CityGaussian" ]]; then
    CITYGAUSSIAN_ROOT="$(cd "$ROOT/../CityGaussian" && pwd)"
  else
    CITYGAUSSIAN_ROOT="$ROOT/third_party/CityGaussian"
  fi
fi
export CITYGAUSSIAN_ROOT
export SMBU_MODEL_DIR="${SMBU_MODEL_DIR:-$ROOT/data_demo/smbu_model}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

if ! python -c "import direct_abs_cost_cuda" 2>/dev/null; then
  echo "Building direct_abs_cost_cuda..."
  (cd DirectAbsoluteCostCuda && python setup_build.py install)
fi
