#!/usr/bin/env bash
set -uo pipefail

GPU="${1:?GPU id required}"
JOB_ID="${2:?job id required}"
CONFIG="${3:?config path required}"

LOG_ROOT="${DEFENSE70_LOG_ROOT:-logs/acm_revision/defense70_r150}"
STATE_ROOT="${DEFENSE70_STATE_ROOT:-logs/acm_revision/defense70_r150/state}"
MIN_FREE_MIB="${DEFENSE70_MIN_FREE_MIB:-23000}"
STARTUP_WAIT="${DEFENSE70_STARTUP_WAIT:-90}"
MAX_RETRIES="${DEFENSE70_MAX_RETRIES:-2}"
RETRY_SLEEP="${DEFENSE70_RETRY_SLEEP:-60}"

mkdir -p "${LOG_ROOT}" "${STATE_ROOT}/done" "${STATE_ROOT}/failed"

if [[ ! -f "${CONFIG}" ]]; then
  echo "[ERROR] config not found: ${CONFIG}" >&2
  exit 2
fi

readarray -t CFG_INFO < <(
python - "${CONFIG}" <<'PY'
import sys, yaml
with open(sys.argv[1], "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
exp = cfg.get("experiment", {})
print(exp.get("population", "target"))
print(exp.get("output_root", ""))
print(cfg.get("federated", {}).get("rounds", ""))
print(cfg.get("defense", {}).get("name", "none"))
PY
)

POPULATION="${CFG_INFO[0]:-target}"
OUTPUT_ROOT="${CFG_INFO[1]:-}"
EXPECTED_ROUNDS="${CFG_INFO[2]:-}"
DEFENSE_NAME="${CFG_INFO[3]:-none}"

if [[ -z "${OUTPUT_ROOT}" ]]; then
  echo "[ERROR] experiment.output_root missing in ${CONFIG}" >&2
  exit 2
fi

DONE_MARKER="${STATE_ROOT}/done/${JOB_ID}.done"
FAIL_MARKER="${STATE_ROOT}/failed/${JOB_ID}.failed"

job_done() {
python - "${OUTPUT_ROOT}" "${EXPECTED_ROUNDS}" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
expected_rounds = int(sys.argv[2]) if sys.argv[2] else None
if not root.exists():
    raise SystemExit(1)
for summary_path in root.rglob("summary.json"):
    run_dir = summary_path.parent
    manifest_path = run_dir / "capture_manifest.json"
    metadata_path = run_dir / "update_metadata.csv"
    if not manifest_path.exists() or not metadata_path.exists() or metadata_path.stat().st_size <= 0:
        continue
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        continue
    if manifest.get("status") != "PASS":
        continue
    if expected_rounds is not None and int(summary.get("rounds", -1)) != expected_rounds:
        continue
    rounds = manifest.get("rounds") or []
    if expected_rounds is not None and (not rounds or max(map(int, rounds)) != expected_rounds):
        continue
    raise SystemExit(0)
raise SystemExit(1)
PY
}

wait_for_gpu_free() {
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  while true; do
    local free_mib
    free_mib="$(nvidia-smi -i "${GPU}" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -dc '0-9')"
    if [[ "${free_mib}" =~ ^[0-9]+$ ]] && (( free_mib >= MIN_FREE_MIB )); then
      echo "[$(date '+%F %T')] GPU=${GPU} ready free=${free_mib} MiB"
      return 0
    fi
    echo "[$(date '+%F %T')] WAIT GPU=${GPU} free=${free_mib:-unknown} MiB; need >= ${MIN_FREE_MIB} MiB"
    sleep 30
  done
}

if job_done; then
  printf '%s\t%s\t%s\n' "$(date '+%F %T')" "${JOB_ID}" "PASS(existing)" > "${DONE_MARKER}"
  rm -f "${FAIL_MARKER}"
  echo "[SKIP PASS] ${JOB_ID}"
  exit 0
fi

LOCK_FILE="/tmp/acm_defense70_gpu${GPU}.startup.lock"
exec {LOCK_FD}>"${LOCK_FILE}"

ATTEMPT=0
while (( ATTEMPT <= MAX_RETRIES )); do
  ATTEMPT=$((ATTEMPT + 1))
  LOG="${LOG_ROOT}/${JOB_ID}.attempt${ATTEMPT}.log"

  # Do not delete or rename OUTPUT_ROOT. Resume checkpoints and finished rounds
  # must remain in place so the next attempt can continue instead of restarting.
  mkdir -p "${OUTPUT_ROOT}"

  echo "[STARTUP LOCK] ${JOB_ID} waiting for GPU=${GPU}"
  flock "${LOCK_FD}"
  wait_for_gpu_free

  echo "============================================================"
  echo "START JOB=${JOB_ID}"
  echo "GPU=${GPU}"
  echo "POPULATION=${POPULATION}"
  echo "ROUNDS=${EXPECTED_ROUNDS}"
  echo "DEFENSE=${DEFENSE_NAME}"
  echo "ATTEMPT=${ATTEMPT}/$((MAX_RETRIES + 1))"
  echo "CONFIG=${CONFIG}"
  echo "LOG=${LOG}"
  echo "============================================================"

  export MFL_NFS_WRITE_RETRIES="${MFL_NFS_WRITE_RETRIES:-8}"

  PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${GPU}" \
    python -m src.acm_revision.privacy_capture_runner_resume_v2 \
      --config "${CONFIG}" \
      --population "${POPULATION}" \
      > "${LOG}" 2>&1 &
  PID=$!

  sleep "${STARTUP_WAIT}"

  if ! kill -0 "${PID}" 2>/dev/null; then
    flock -u "${LOCK_FD}"
    wait "${PID}" || true
    echo "[FAILED DURING STARTUP] ${JOB_ID}" >&2
    tail -n 160 "${LOG}" >&2 || true
  else
    FREE_AFTER="$(nvidia-smi -i "${GPU}" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -dc '0-9')"
    echo "[STARTED] ${JOB_ID}; GPU=${GPU}; free_after=${FREE_AFTER:-unknown} MiB"
    flock -u "${LOCK_FD}"

    STATUS=0
    wait "${PID}" || STATUS=$?

    if (( STATUS == 0 )) && job_done; then
      printf '%s\t%s\t%s\n' "$(date '+%F %T')" "${JOB_ID}" "PASS" > "${DONE_MARKER}"
      rm -f "${FAIL_MARKER}"
      find "${OUTPUT_ROOT}" -type d -name resume -prune -exec rm -rf {} + 2>/dev/null || true
      find "${OUTPUT_ROOT}" -type f -name best_model.pt -delete 2>/dev/null || true
      echo "[DONE PASS] ${JOB_ID}"
      exit 0
    fi

    echo "[FAILED] ${JOB_ID}; exit=${STATUS}" >&2
    tail -n 200 "${LOG}" >&2 || true
  fi

  if (( ATTEMPT <= MAX_RETRIES )); then
    echo "[RETRY/RESUME] ${JOB_ID} in ${RETRY_SLEEP}s"
    sleep "${RETRY_SLEEP}"
  fi
done

printf '%s\t%s\t%s\n' "$(date '+%F %T')" "${JOB_ID}" "FAILED_AFTER_RETRIES" > "${FAIL_MARKER}"
echo "[GIVE UP] ${JOB_ID} after $((MAX_RETRIES + 1)) attempts" >&2
echo "Re-run the same server launcher later; the job will resume from its newest valid round checkpoint." >&2
exit 1
