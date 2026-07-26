#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-}"
GPU_LIST="${2:-0}"
MATRIX="${MATRIX:-configs/acm_revision/experiment_matrix.yaml}"
GENERATED_ROOT="${GENERATED_ROOT:-configs/acm_revision/generated}"
LOG_ROOT="${LOG_ROOT:-logs/acm_revision}"

if [[ -z "$PROFILE" ]]; then
  echo "Usage: $0 <profile-name> [comma-separated-gpu-ids]" >&2
  exit 2
fi

python -m src.acm_revision.cli generate-jobs \
  --matrix "$MATRIX" \
  --profile "$PROFILE" \
  --output-dir "$GENERATED_ROOT"

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
mapfile -t CONFIGS < <(find "$GENERATED_ROOT/$PROFILE" -maxdepth 1 -name '*.yaml' | sort)
mkdir -p "$LOG_ROOT/$PROFILE"

run_one() {
  local config="$1"
  local gpu="$2"
  local name
  name="$(basename "$config" .yaml)"
  echo "[$(date '+%F %T')] START $name GPU=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" python -m src.acm_revision.cli run-fl --config "$config" \
    > "$LOG_ROOT/$PROFILE/${name}.log" 2>&1
  echo "[$(date '+%F %T')] DONE  $name GPU=$gpu"
}

active=0
for idx in "${!CONFIGS[@]}"; do
  gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
  run_one "${CONFIGS[$idx]}" "$gpu" &
  active=$((active + 1))
  if (( active >= ${#GPUS[@]} )); then
    wait -n
    active=$((active - 1))
  fi
done
wait
