#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <queue.tsv> <gpu_id> [log_dir]"
  exit 2
fi

QUEUE="$1"
GPU_ID="$2"
LOG_DIR="${3:-logs/headfix}"
mkdir -p "$LOG_DIR"

if [[ ! -f "$QUEUE" ]]; then
  echo "Queue file not found: $QUEUE" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1

echo "[HEADFIX QUEUE] queue=$QUEUE gpu=$CUDA_VISIBLE_DEVICES"
echo "[HEADFIX QUEUE] started=$(date -Iseconds)"

tail -n +2 "$QUEUE" | while IFS=$'\t' read -r job_id config_path population seed; do
  [[ -z "${job_id:-}" ]] && continue
  log="$LOG_DIR/${job_id}.log"
  echo "[HEADFIX START] job=$job_id population=$population seed=$seed config=$config_path"
  python -m src.acm_revision.privacy_capture_runner_resume_v2 \
    --config "$config_path" \
    --population "$population" \
    2>&1 | tee "$log"
  echo "[HEADFIX DONE] job=$job_id"
done

echo "[HEADFIX QUEUE] finished=$(date -Iseconds)"
