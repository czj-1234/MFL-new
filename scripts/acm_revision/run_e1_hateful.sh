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
  if [[ ${#GPUS[@]} -eq 2 ]]; then
    echo "[INFO] Seed split: GPU ${GPUS[0]} <- seeds 42-46; GPU ${GPUS[1]} <- seeds 47-51"
    echo "[INFO] Each GPU runs 5 seeds x 4 settings x 5 concentrations = 100 jobs sequentially."
  fi
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

config_seed() {
  local config="$1"
  awk '/^seed:[[:space:]]*/ {print $2; exit}' "${config}"
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

  echo "[$(date '+%F %T')] START ${job_id} GPU=${gpu} seed=$(config_seed "${config}")"
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

run_seed_worker() {
  local gpu="$1"
  local seed_min="$2"
  local seed_max="$3"
  local failures=0
  local config seed

  echo "[$(date '+%F %T')] WORKER GPU=${gpu} seeds=${seed_min}-${seed_max}"
  for config in "${CONFIGS[@]}"; do
    seed="$(config_seed "${config}")"
    if [[ -z "${seed}" ]]; then
      echo "[ERROR] Could not read seed from ${config}" >&2
      failures=$((failures + 1))
      continue
    fi
    if (( seed < seed_min || seed > seed_max )); then
      continue
    fi
    if ! run_one "${config}" "${gpu}"; then
      failures=$((failures + 1))
    fi
  done

  if (( failures > 0 )); then
    echo "[ERROR] GPU ${gpu} worker had ${failures} failed job(s)." >&2
    return 1
  fi
  echo "[$(date '+%F %T')] WORKER DONE GPU=${gpu} seeds=${seed_min}-${seed_max}"
}

# Smoke mode remains one formal job on the first GPU.
if [[ "${MODE}" == "smoke" ]]; then
  run_one "${CONFIGS[0]}" "${GPUS[0]}"
  exit $?
fi

# Preferred two-GPU protocol: split independent FL seeds 5 + 5.
if [[ ${#GPUS[@]} -eq 2 ]]; then
  failures=0
  run_seed_worker "${GPUS[0]}" 42 46 &
  pid0=$!
  run_seed_worker "${GPUS[1]}" 47 51 &
  pid1=$!

  if ! wait "${pid0}"; then
    failures=$((failures + 1))
  fi
  if ! wait "${pid1}"; then
    failures=$((failures + 1))
  fi

  if (( failures > 0 )); then
    echo "[ERROR] ${failures} GPU worker(s) reported failures. Re-run the same command; completed jobs will be skipped." >&2
    exit 1
  fi

  echo "[OK] Full E1 Hateful run finished successfully."
  exit 0
fi

# Generic fallback for one GPU or more than two GPUs: one job at a time per GPU.
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

echo "[OK] Full E1 Hateful run finished successfully."
