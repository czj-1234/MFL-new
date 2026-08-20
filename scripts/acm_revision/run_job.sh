#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <resolved-config.yaml> [gpu-id]" >&2
  exit 2
fi

CONFIG="$1"
GPU_ID="${2:-}"

if [[ -n "$GPU_ID" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

python -m src.acm_revision.cli run-fl --config "$CONFIG"
