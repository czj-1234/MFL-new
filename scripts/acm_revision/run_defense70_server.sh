#!/usr/bin/env bash
set -uo pipefail

SERVER="${1:?server id A or B required}"
GPU0="${2:-0}"
GPU1="${3:-1}"
SERVER="${SERVER^^}"

JOBS_FILE="${DEFENSE70_JOBS_FILE:-configs/acm_revision/generated/defense70/jobs.tsv}"
JOB_SCRIPT="scripts/acm_revision/run_defense70_job.sh"
STATE_ROOT="${DEFENSE70_STATE_ROOT:-logs/acm_revision/defense70_r150/state}"

if [[ "${SERVER}" != "A" && "${SERVER}" != "B" ]]; then
  echo "SERVER must be A or B" >&2
  exit 2
fi
if [[ "${GPU0}" == "${GPU1}" ]]; then
  echo "GPU ids must be different" >&2
  exit 2
fi
[[ -f "${JOBS_FILE}" ]] || { echo "[ERROR] Missing ${JOBS_FILE}" >&2; exit 2; }
[[ -x "${JOB_SCRIPT}" ]] || { echo "[ERROR] Missing/executable ${JOB_SCRIPT}" >&2; exit 2; }

mkdir -p "${STATE_ROOT}/server${SERVER}"

declare -a Q0A=() Q0B=() Q1A=() Q1B=()
GLOBAL_IDX=0
LOCAL_IDX=0

while IFS=$'\t' read -r JOB_ID CONFIG_PATH EXTRA; do
  [[ -z "${JOB_ID:-}" ]] && continue
  [[ "${JOB_ID:0:1}" == "#" ]] && continue
  [[ "${JOB_ID}" == "job_id" ]] && continue
  [[ -z "${EXTRA:-}" ]] || { echo "[ERROR] jobs.tsv must have 2 columns" >&2; exit 2; }
  [[ -f "${CONFIG_PATH}" ]] || { echo "[ERROR] Missing config ${CONFIG_PATH}" >&2; exit 2; }

  USE=0
  [[ "${SERVER}" == "A" && $((GLOBAL_IDX % 2)) -eq 0 ]] && USE=1
  [[ "${SERVER}" == "B" && $((GLOBAL_IDX % 2)) -eq 1 ]] && USE=1

  if (( USE == 1 )); then
    ITEM="${JOB_ID}"$'\t'"${CONFIG_PATH}"
    case $((LOCAL_IDX % 4)) in
      0) Q0A+=("${ITEM}") ;;
      1) Q0B+=("${ITEM}") ;;
      2) Q1A+=("${ITEM}") ;;
      3) Q1B+=("${ITEM}") ;;
    esac
    LOCAL_IDX=$((LOCAL_IDX + 1))
  fi
  GLOBAL_IDX=$((GLOBAL_IDX + 1))
done < "${JOBS_FILE}"

(( GLOBAL_IDX == 70 )) || { echo "[ERROR] Expected 70 jobs, got ${GLOBAL_IDX}" >&2; exit 2; }
(( LOCAL_IDX == 35 )) || { echo "[ERROR] Server ${SERVER} expected 35 jobs, got ${LOCAL_IDX}" >&2; exit 2; }

echo "============================================================"
echo "DEFENSE70 SERVER ${SERVER}"
echo "GPU ${GPU0} slot0=${#Q0A[@]} slot1=${#Q0B[@]}"
echo "GPU ${GPU1} slot0=${#Q1A[@]} slot1=${#Q1B[@]}"
echo "Each run uses rolling round-level checkpoints and resumes automatically."
echo "============================================================"

run_worker() {
  local GPU="$1" SLOT="$2"
  shift 2
  local ITEM JOB_ID CONFIG_PATH failures=0
  for ITEM in "$@"; do
    IFS=$'\t' read -r JOB_ID CONFIG_PATH <<< "${ITEM}"
    echo "[SERVER ${SERVER}] GPU=${GPU} SLOT=${SLOT} JOB=${JOB_ID}"
    if ! bash "${JOB_SCRIPT}" "${GPU}" "${JOB_ID}" "${CONFIG_PATH}"; then
      failures=$((failures + 1))
      echo "[WARNING] ${JOB_ID} failed after retries; continuing queue." >&2
    fi
  done
  echo "${failures}" > "${STATE_ROOT}/server${SERVER}/gpu${GPU}_slot${SLOT}.failcount"
  (( failures == 0 ))
}

run_worker "${GPU0}" 0 "${Q0A[@]}" & PID0A=$!
sleep 8
run_worker "${GPU0}" 1 "${Q0B[@]}" & PID0B=$!
sleep 8
run_worker "${GPU1}" 0 "${Q1A[@]}" & PID1A=$!
sleep 8
run_worker "${GPU1}" 1 "${Q1B[@]}" & PID1B=$!

cleanup() {
  kill -TERM "${PID0A}" "${PID0B}" "${PID1A}" "${PID1B}" 2>/dev/null || true
}
trap cleanup INT TERM

S0A=0; S0B=0; S1A=0; S1B=0
wait "${PID0A}" || S0A=$?
wait "${PID0B}" || S0B=$?
wait "${PID1A}" || S1A=$?
wait "${PID1B}" || S1B=$?
trap - INT TERM

if (( S0A != 0 || S0B != 0 || S1A != 0 || S1B != 0 )); then
  echo "[DONE WITH FAILURES] Re-run same launcher; completed jobs skip and interrupted jobs resume." >&2
  exit 1
fi

echo "[DONE PASS] Server ${SERVER}: all 35 jobs passed."
