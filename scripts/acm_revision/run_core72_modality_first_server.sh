#!/usr/bin/env bash
set -euo pipefail

SERVER="${1:?server id A or B required}"
GPU0="${2:-0}"
GPU1="${3:-1}"
ROUNDS=150
NAMESPACE="core72_modality_first_v2"
PLAN_MODULE="src.acm_revision.core72_modality_first_plan"
JOB_SCRIPT="scripts/acm_revision/run_core72_job.sh"
SMOKE_SCRIPT="scripts/acm_revision/smoke_test_core72_modality_first.sh"
RESULT_ROOT="results/acm_revision/${NAMESPACE}_r${ROUNDS}"
POST_ROOT="results/acm_revision/${NAMESPACE}_postprocess_r${ROUNDS}"
CONFIG_ROOT="configs/acm_revision/generated/${NAMESPACE}_r${ROUNDS}"
LOG_ROOT="logs/acm_revision/${NAMESPACE}_r${ROUNDS}"
PREFLIGHT_ROOT="results/acm_revision/${NAMESPACE}_preflight"
SERVER="${SERVER^^}"
PREFLIGHT="${PREFLIGHT_ROOT}/server${SERVER}.json"

if [[ "${SERVER}" != "A" && "${SERVER}" != "B" ]]; then
  echo "SERVER must be A or B" >&2
  exit 2
fi
if [[ "${GPU0}" == "${GPU1}" ]]; then
  echo "GPU ids must be different" >&2
  exit 2
fi
for path in "${JOB_SCRIPT}" "${SMOKE_SCRIPT}"; do
  [[ -f "${path}" ]] || { echo "Missing ${path}; run from repository root." >&2; exit 2; }
done

# Hard separation from every earlier experiment namespace.
case "${RESULT_ROOT}" in
  *core72_modality_first_v2*) ;;
  *) echo "Unsafe result namespace: ${RESULT_ROOT}" >&2; exit 2 ;;
esac
mkdir -p "${RESULT_ROOT}" "${POST_ROOT}" "${CONFIG_ROOT}" "${LOG_ROOT}" "${PREFLIGHT_ROOT}"
for path in "${RESULT_ROOT}" "${POST_ROOT}" "${CONFIG_ROOT}" "${LOG_ROOT}"; do
  [[ -w "${path}" ]] || { echo "Path is not writable: ${path}" >&2; exit 2; }
done

python -m "${PLAN_MODULE}" validate \
  --check-data \
  --output "${PREFLIGHT}"

if [[ "${RUN_CORE72_SMOKE:-1}" == "1" ]]; then
  bash "${SMOKE_SCRIPT}" "${GPU0}"
fi

mapfile -t QUEUE0 < <(python -m "${PLAN_MODULE}" jobs --server "${SERVER}" --queue 0)
mapfile -t QUEUE1 < <(python -m "${PLAN_MODULE}" jobs --server "${SERVER}" --queue 1)

if (( ${#QUEUE0[@]} == 0 || ${#QUEUE1[@]} == 0 )); then
  echo "One or both GPU queues are empty." >&2
  exit 2
fi
for first in "${QUEUE0[0]}" "${QUEUE1[0]}"; do
  IFS=$'\t' read -r first_setting _ <<< "${first}"
  if [[ "${first_setting}" != "modality_exclusive" ]]; then
    echo "Safety check failed: every GPU queue must start with modality_exclusive; got ${first_setting}" >&2
    exit 2
  fi
done

TOTAL=$(( ${#QUEUE0[@]} + ${#QUEUE1[@]} ))
echo "============================================================"
echo "CORE72 MODALITY-FIRST V2 — SERVER ${SERVER}"
echo "New result root: ${RESULT_ROOT}"
echo "Previous Core72/diagnostic result folders are not read or overwritten."
echo "GPU ${GPU0}: ${#QUEUE0[@]} jobs; first=${QUEUE0[0]}"
echo "GPU ${GPU1}: ${#QUEUE1[@]} jobs; first=${QUEUE1[0]}"
echo "Server total: ${TOTAL} jobs"
echo "Every job: 150 rounds; exact float32 head updates every round."
echo "============================================================"

run_queue() {
  local gpu="$1"
  shift
  local line setting concentration population seed rounds num_clients samples job_id
  local seen_non_modality=0
  for line in "$@"; do
    IFS=$'\t' read -r setting concentration population seed rounds num_clients samples job_id <<< "${line}"
    if [[ "${setting}" != "modality_exclusive" ]]; then
      seen_non_modality=1
    elif (( seen_non_modality == 1 )); then
      echo "Queue-order violation on GPU ${gpu}: modality_exclusive appeared after a single-modality job." >&2
      return 2
    fi
    echo "[SERVER ${SERVER}] GPU=${gpu} ${job_id}"
    CORE72_RESULT_ROOT="${RESULT_ROOT}" \
    CORE72_CONFIG_ROOT="${CONFIG_ROOT}" \
    CORE72_LOG_ROOT="${LOG_ROOT}" \
    CORE72_NUM_CLIENTS="${num_clients}" \
    CORE72_SAMPLES_PER_CLIENT="${samples}" \
      bash "${JOB_SCRIPT}" "${gpu}" "${setting}" "${concentration}" "${population}" "${seed}" "${rounds}"
  done
}

run_queue "${GPU0}" "${QUEUE0[@]}" &
PID0=$!
run_queue "${GPU1}" "${QUEUE1[@]}" &
PID1=$!
cleanup() { kill -TERM "${PID0}" "${PID1}" 2>/dev/null || true; }
trap cleanup INT TERM
STATUS0=0
STATUS1=0
wait "${PID0}" || STATUS0=$?
wait "${PID1}" || STATUS1=$?
trap - INT TERM
if (( STATUS0 != 0 || STATUS1 != 0 )); then
  echo "[FAIL] Server ${SERVER} queues exited with ${STATUS0}/${STATUS1}" >&2
  exit 1
fi

if [[ "${RUN_CORE72_POSTPROCESS:-1}" == "1" ]]; then
  mapfile -t CELLS < <(python -m "${PLAN_MODULE}" cells --server "${SERVER}")
  for cell in "${CELLS[@]}"; do
    IFS=$'\t' read -r setting concentration <<< "${cell}"
    echo "[POSTPROCESS] server=${SERVER} setting=${setting} concentration=${concentration}"
    python -m src.acm_revision.core72_postprocess \
      --root "${RESULT_ROOT}" \
      --setting "${setting}" \
      --concentration "${concentration}" \
      --rounds "${ROUNDS}" \
      --output-root "${POST_ROOT}"
    ctag="${concentration//./p}"
    output_dir="${POST_ROOT}/${setting}__c${ctag}__r${ROUNDS}"
    python -m src.acm_revision.core72_validate --output-dir "${output_dir}"
  done
fi

echo "[DONE] CORE72 MODALITY-FIRST V2 Server ${SERVER}: training, post-processing and validation passed."
