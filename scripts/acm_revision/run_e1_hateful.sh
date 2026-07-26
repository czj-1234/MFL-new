#!/usr/bin/env bash
set -uo pipefail

GPU_LIST="${1:-0}"
MODE="${2:-all}"   # all | smoke
MATRIX="configs/acm_revision/e1_hateful.yaml"
PROFILE="E1_task_performance_hateful"
GENERATED_ROOT="configs/acm_revision/generated"
LOG_ROOT="logs/acm_revision/${PROFILE}"
RESULT_ROOT="results/acm_revision/${PROFILE}"

if [[ "${MODE}" != "all" && "${MODE}" != "smoke" ]]; then
  echo "Usage: $0 [comma-separated-gpu-ids] [all|smoke]" >&2
  exit 2
fi

python -m src.acm_revision.cli generate-jobs \
  --matrix "${MATRIX}" \
  --profile "${PROFILE}" \
  --output-dir "${GENERATED_ROOT}"

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
if [[ ${#GPUS[@]} -eq 0 ]]; then
  echo "[ERROR] No GPUs supplied." >&2
  exit 2
fi

mapfile -t CONFIGS < <(find "${GENERATED_ROOT}/${PROFILE}" -maxdepth 1 -name '*.yaml' | sort)
mkdir -p "${LOG_ROOT}" "${RESULT_ROOT}"

if [[ ${#CONFIGS[@]} -ne 200 ]]; then
  echo "[ERROR] Expected exactly 200 Hateful E1 jobs, found ${#CONFIGS[@]}." >&2
  exit 1
fi

if [[ "${MODE}" == "smoke" ]]; then
  CONFIGS=("${CONFIGS[0]}")
  GPUS=("${GPUS[0]}")
  echo "[INFO] Smoke mode: running only ${CONFIGS[0]} on GPU ${GPUS[0]}"
else
  echo "[INFO] Full E1 Hateful run: ${#CONFIGS[@]} jobs across GPU(s): ${GPU_LIST}"
fi

job_done() {
  local config="$1"
  local job_id
  job_id="$(basename "${config}" .yaml)"
  local root="${RESULT_ROOT}/${job_id}"
  [[ -d "${root}" ]] && find "${root}" -name summary.json -print -quit 2>/dev/null | grep -q .
}

cleanup_heavy_artifacts() {
  local config="$1"
  local job_id
  job_id="$(basename "${config}" .yaml)"
  local root="${RESULT_ROOT}/${job_id}"
  [[ -d "${root}" ]] || return 0
  find "${root}" -type f -name 'best_model.pt' -delete 2>/dev/null || true
  find "${root}" -type d -name checkpoints -prune -exec rm -rf {} + 2>/dev/null || true
  # E1 has groups: [], so update NPZ files contain no attack vectors and are not needed.
  find "${root}" -type d -name updates -prune -exec rm -rf {} + 2>/dev/null || true
}

run_one() {
  local config="$1"
  local gpu="$2"
  local job_id
  job_id="$(basename "${config}" .yaml)"
  local log="${LOG_ROOT}/${job_id}.log"

  if job_done "${config}"; then
    echo "[$(date '+%F %T')] SKIP  ${job_id} already complete"
    cleanup_heavy_artifacts "${config}"
    return 0
  fi

  echo "[$(date '+%F %T')] START ${job_id} GPU=${gpu}"
  if CUDA_VISIBLE_DEVICES="${gpu}" python -m src.acm_revision.cli run-fl --config "${config}" > "${log}" 2>&1; then
    if job_done "${config}"; then
      cleanup_heavy_artifacts "${config}"
      echo "[$(date '+%F %T')] DONE  ${job_id} GPU=${gpu}"
      return 0
    fi
    echo "[$(date '+%F %T')] FAIL  ${job_id}: process exited 0 but summary.json is missing" >&2
  else
    echo "[$(date '+%F %T')] FAIL  ${job_id} GPU=${gpu}" >&2
  fi

  echo "---------------- tail ${log} ----------------" >&2
  tail -n 60 "${log}" >&2 || true
  echo "------------------------------------------------" >&2
  return 1
}

active=0
failures=0

for idx in "${!CONFIGS[@]}"; do
  config="${CONFIGS[$idx]}"
  gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"

  run_one "${config}" "${gpu}" &
  active=$((active + 1))

  if (( active >= ${#GPUS[@]} )); then
    if ! wait -n; then
      failures=$((failures + 1))
    fi
    active=$((active - 1))
  fi
done

while (( active > 0 )); do
  if ! wait -n; then
    failures=$((failures + 1))
  fi
  active=$((active - 1))
done

if (( failures > 0 )); then
  echo "[ERROR] ${failures} job(s) failed. Re-run the same command after fixing the first error; completed jobs will be skipped." >&2
  exit 1
fi

echo "[OK] ${MODE} run finished successfully."
