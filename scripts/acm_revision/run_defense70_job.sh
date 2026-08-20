#!/usr/bin/env bash
set -uo pipefail

GPU="${1:?GPU id required}"
JOB_ID="${2:?job id required}"
CONFIG="${3:?config path required}"

LOG_ROOT="${DEFENSE70_LOG_ROOT:-logs/acm_revision/defense70_r150}"
STATE_ROOT="${DEFENSE70_STATE_ROOT:-logs/acm_revision/defense70_r150/state}"
CHECKPOINT_ROOT="${DEFENSE70_CHECKPOINT_ROOT:-checkpoints/acm_revision/defense70_r150}"
RUNTIME_CFG_ROOT="${DEFENSE70_RUNTIME_CFG_ROOT:-logs/acm_revision/defense70_r150/runtime_configs}"
CHECKPOINT_EVERY="${DEFENSE70_CHECKPOINT_EVERY:-15}"
CHECKPOINT_KEEP_LAST="${DEFENSE70_CHECKPOINT_KEEP_LAST:-2}"
MIN_FREE_MIB="${DEFENSE70_MIN_FREE_MIB:-23000}"
STARTUP_WAIT="${DEFENSE70_STARTUP_WAIT:-90}"
MAX_RETRIES="${DEFENSE70_MAX_RETRIES:-2}"
RETRY_SLEEP="${DEFENSE70_RETRY_SLEEP:-60}"

mkdir -p "${LOG_ROOT}" "${STATE_ROOT}/done" "${STATE_ROOT}/failed" "${CHECKPOINT_ROOT}" "${RUNTIME_CFG_ROOT}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "[ERROR] config not found: ${CONFIG}" >&2
  exit 2
fi

if ! [[ "${CHECKPOINT_EVERY}" =~ ^[0-9]+$ ]] || (( CHECKPOINT_EVERY <= 0 )); then
  echo "[ERROR] DEFENSE70_CHECKPOINT_EVERY must be a positive integer" >&2
  exit 2
fi
if ! [[ "${CHECKPOINT_KEEP_LAST}" =~ ^[0-9]+$ ]] || (( CHECKPOINT_KEEP_LAST <= 0 )); then
  echo "[ERROR] DEFENSE70_CHECKPOINT_KEEP_LAST must be a positive integer" >&2
  exit 2
fi

readarray -t CFG_INFO < <(
python - "${CONFIG}" <<'PY'
import sys, yaml
with open(sys.argv[1], "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
exp = cfg.get("experiment", {})
fed = cfg.get("federated", {})
population = exp.get("population", "target")
model = cfg.get("model", {}).get("architecture", "clip_dual")
defense = cfg.get("defense", {}).get("name", "none")
concentration = exp.get("concentration", exp.get("association", "iid"))
run_id = (
    f"{cfg.get('data', {}).get('name','dataset')}__{model}__{population}__{exp['setting_name']}__c{concentration}"
    f"__n{fed['num_clients']}__p{fed.get('participation_rate',1.0)}__{fed.get('aggregation','fedavg')}"
    f"__seed{cfg['seed']}__def-{defense}"
).replace("/", "-")
print(population)
print(exp.get("output_root", ""))
print(fed.get("rounds", ""))
print(defense)
print(run_id)
PY
)

POPULATION="${CFG_INFO[0]:-target}"
OUTPUT_ROOT="${CFG_INFO[1]:-}"
EXPECTED_ROUNDS="${CFG_INFO[2]:-}"
DEFENSE_NAME="${CFG_INFO[3]:-none}"
RUN_ID="${CFG_INFO[4]:-}"

if [[ -z "${OUTPUT_ROOT}" ]]; then
  echo "[ERROR] experiment.output_root missing in ${CONFIG}" >&2
  exit 2
fi
if [[ -z "${RUN_ID}" ]]; then
  echo "[ERROR] could not determine run_id from ${CONFIG}" >&2
  exit 2
fi

RESULT_RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}"
CHECKPOINT_JOB_ROOT="${CHECKPOINT_ROOT}/${JOB_ID}"
RESUME_LINK="${RESULT_RUN_DIR}/resume"
BEST_LINK="${RESULT_RUN_DIR}/best_model.pt"
BEST_TARGET="${CHECKPOINT_JOB_ROOT}/best_model.pt"
RUNTIME_CONFIG="${RUNTIME_CFG_ROOT}/${JOB_ID}.yaml"
DONE_MARKER="${STATE_ROOT}/done/${JOB_ID}.done"
FAIL_MARKER="${STATE_ROOT}/failed/${JOB_ID}.failed"

# Create a runtime copy so every formal Defense70 run uses true round resume
# every 15 rounds by default, without changing the frozen scientific YAML.
python - "${CONFIG}" "${RUNTIME_CONFIG}" "${CHECKPOINT_EVERY}" "${CHECKPOINT_KEEP_LAST}" <<'PY'
import copy, sys, yaml
src, dst, every, keep = sys.argv[1:]
with open(src, "r", encoding="utf-8") as f:
    cfg = copy.deepcopy(yaml.safe_load(f))
cfg["resume"] = {
    "enabled": True,
    "checkpoint_every": int(every),
    "keep_last": int(keep),
}
# Formal Defense70 uses the separate rolling resume checkpoint system.
# Do not also create ordinary model checkpoints inside results/.
uc = cfg.setdefault("update_capture", {})
uc["save_all_checkpoints"] = False
uc["checkpoint_rounds"] = []
uc["model_checkpoint_rounds"] = []
with open(dst, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY

job_done() {
python - "${RESULT_RUN_DIR}" "${EXPECTED_ROUNDS}" <<'PY'
import json, pathlib, sys
run_dir = pathlib.Path(sys.argv[1])
expected_rounds = int(sys.argv[2]) if sys.argv[2] else None
summary_path = run_dir / "summary.json"
manifest_path = run_dir / "capture_manifest.json"
metadata_path = run_dir / "update_metadata.csv"
if not summary_path.exists() or not manifest_path.exists() or not metadata_path.exists() or metadata_path.stat().st_size <= 0:
    raise SystemExit(1)
try:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)
if manifest.get("status") != "PASS":
    raise SystemExit(1)
if expected_rounds is not None and int(summary.get("rounds", -1)) != expected_rounds:
    raise SystemExit(1)
rounds = manifest.get("rounds") or []
if expected_rounds is not None and (not rounds or max(map(int, rounds)) != expected_rounds):
    raise SystemExit(1)
raise SystemExit(0)
PY
}

prepare_checkpoint_links() {
  mkdir -p "${RESULT_RUN_DIR}" "${CHECKPOINT_JOB_ROOT}"

  local desired_resume
  desired_resume="$(python - "${CHECKPOINT_JOB_ROOT}" <<'PY'
import os, sys
print(os.path.abspath(sys.argv[1]))
PY
)"

  if [[ -L "${RESUME_LINK}" ]]; then
    local current_resume
    current_resume="$(readlink -f "${RESUME_LINK}" 2>/dev/null || true)"
    if [[ "${current_resume}" != "${desired_resume}" ]]; then
      echo "[ERROR] resume symlink points elsewhere: ${RESUME_LINK} -> ${current_resume}" >&2
      return 2
    fi
  elif [[ -d "${RESUME_LINK}" ]]; then
    # Migration path for a smoke run made before checkpoints were separated.
    echo "[MIGRATE] moving existing resume checkpoints out of results: ${RESUME_LINK} -> ${CHECKPOINT_JOB_ROOT}"
    cp -a "${RESUME_LINK}/." "${CHECKPOINT_JOB_ROOT}/" 2>/dev/null || true
    rm -rf "${RESUME_LINK}"
    ln -s "${desired_resume}" "${RESUME_LINK}"
  elif [[ -e "${RESUME_LINK}" ]]; then
    echo "[ERROR] ${RESUME_LINK} exists but is not a directory/symlink" >&2
    return 2
  else
    ln -s "${desired_resume}" "${RESUME_LINK}"
  fi

  local desired_best
  desired_best="$(python - "${BEST_TARGET}" <<'PY'
import os, sys
print(os.path.abspath(sys.argv[1]))
PY
)"

  if [[ -L "${BEST_LINK}" ]]; then
    local current_best
    current_best="$(readlink -f "${BEST_LINK}" 2>/dev/null || true)"
    if [[ -n "${current_best}" && "${current_best}" != "${desired_best}" ]]; then
      echo "[ERROR] best_model symlink points elsewhere: ${BEST_LINK} -> ${current_best}" >&2
      return 2
    fi
  elif [[ -f "${BEST_LINK}" ]]; then
    echo "[MIGRATE] moving best_model.pt out of results"
    mv "${BEST_LINK}" "${BEST_TARGET}"
    ln -s "${desired_best}" "${BEST_LINK}"
  elif [[ -e "${BEST_LINK}" ]]; then
    echo "[ERROR] ${BEST_LINK} exists but is not a file/symlink" >&2
    return 2
  else
    ln -s "${desired_best}" "${BEST_LINK}"
  fi
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

prepare_checkpoint_links || exit $?

LOCK_FILE="/tmp/acm_defense70_gpu${GPU}.startup.lock"
exec {LOCK_FD}>"${LOCK_FILE}"

ATTEMPT=0
while (( ATTEMPT <= MAX_RETRIES )); do
  ATTEMPT=$((ATTEMPT + 1))
  LOG="${LOG_ROOT}/${JOB_ID}.attempt${ATTEMPT}.log"

  # Never delete RESULT_RUN_DIR or CHECKPOINT_JOB_ROOT here. An interrupted
  # run must retain both completed-round results and its external checkpoints.
  mkdir -p "${RESULT_RUN_DIR}" "${CHECKPOINT_JOB_ROOT}"

  echo "[STARTUP LOCK] ${JOB_ID} waiting for GPU=${GPU}"
  flock "${LOCK_FD}"
  wait_for_gpu_free

  echo "============================================================"
  echo "START JOB=${JOB_ID}"
  echo "GPU=${GPU}"
  echo "POPULATION=${POPULATION}"
  echo "ROUNDS=${EXPECTED_ROUNDS}"
  echo "DEFENSE=${DEFENSE_NAME}"
  echo "CHECKPOINT_EVERY=${CHECKPOINT_EVERY}"
  echo "RESULT_DIR=${RESULT_RUN_DIR}"
  echo "CHECKPOINT_DIR=${CHECKPOINT_JOB_ROOT}"
  echo "ATTEMPT=${ATTEMPT}/$((MAX_RETRIES + 1))"
  echo "CONFIG=${RUNTIME_CONFIG}"
  echo "LOG=${LOG}"
  echo "============================================================"

  export MFL_NFS_WRITE_RETRIES="${MFL_NFS_WRITE_RETRIES:-8}"

  PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${GPU}" \
    python -m src.acm_revision.privacy_capture_runner_resume_v2 \
      --config "${RUNTIME_CONFIG}" \
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
      # Keep physical checkpoints in checkpoints/... but remove links from
      # results/... so the result tree contains results only after completion.
      [[ -L "${RESUME_LINK}" ]] && rm -f "${RESUME_LINK}"
      [[ -L "${BEST_LINK}" ]] && rm -f "${BEST_LINK}"
      echo "[DONE PASS] ${JOB_ID}"
      echo "[CHECKPOINTS KEPT] ${CHECKPOINT_JOB_ROOT}"
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
echo "Re-run the same server launcher later; the job will resume from its newest valid checkpoint in ${CHECKPOINT_JOB_ROOT}." >&2
exit 1
