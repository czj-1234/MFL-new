#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"
LOG_DIR="${2:-logs/headfix}"
CONFIG="configs/acm_revision/generated/headfix/headfix_06__target__seed45.yaml"
JOB_ID="headfix_06__target__seed45"

mkdir -p "$LOG_DIR"

if [[ ! -f "$CONFIG" ]]; then
  echo "Config file not found: $CONFIG" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1

echo "[HEADFIX START] job=$JOB_ID population=target seed=45 gpu=$CUDA_VISIBLE_DEVICES"
echo "[HEADFIX START] started=$(date -Iseconds)"

python -m src.acm_revision.privacy_capture_runner_resume_v2 \
  --config "$CONFIG" \
  --population target \
  2>&1 | tee "$LOG_DIR/${JOB_ID}.log"

echo "[HEADFIX DONE] job=$JOB_ID finished=$(date -Iseconds)"
