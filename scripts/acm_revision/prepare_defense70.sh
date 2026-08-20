#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
PREP_ROOT="${DEFENSE70_PREP_ROOT:-results/acm_revision/defense70_prep}"
STATUS_DIR="${DEFENSE70_PREP_STATUS_DIR:-logs/acm_revision/defense70_r150/prep}"
GEN_DIR="${DEFENSE70_CONFIG_ROOT:-configs/acm_revision/generated/defense70}"
PREP_CFG_DIR="configs/acm_revision/generated/defense70_prep"
REF_CFG="${PREP_CFG_DIR}/reference_seed142.yaml"
MANIFEST="${PREP_ROOT}/locked_params.yaml"
JOBS_FILE="${GEN_DIR}/jobs.tsv"
LOCK_FILE="${STATUS_DIR}/prepare.lock"
LOG_FILE="${STATUS_DIR}/prepare.log"

mkdir -p "${PREP_ROOT}" "${STATUS_DIR}" "${GEN_DIR}" "${PREP_CFG_DIR}"

status() {
  local stage="$1"
  shift
  local message="$*"
  python - "${STATUS_DIR}/status.json" "${stage}" "${message}" <<'PY'
import json, pathlib, sys, time
path, stage, message = sys.argv[1:]
obj = {"stage": stage, "message": message, "updated_unix": time.time()}
p = pathlib.Path(path)
p.parent.mkdir(parents=True, exist_ok=True)
tmp = p.with_suffix(".tmp")
tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
tmp.replace(p)
PY
  echo "[$(date '+%F %T')] [DEFENSE70 PREP] ${stage}: ${message}" | tee -a "${LOG_FILE}"
}

on_error() {
  local rc=$?
  set +e
  status "FAILED" "preparation exited with code ${rc}; inspect ${LOG_FILE}"
  exit "${rc}"
}
trap on_error ERR

ready_check() {
  [[ -f "${MANIFEST}" && -f "${JOBS_FILE}" ]] || return 1
  python - "${MANIFEST}" "${JOBS_FILE}" <<'PY'
import pathlib, sys, yaml
manifest_path, jobs_path = map(pathlib.Path, sys.argv[1:])
manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
if manifest.get("status") != "LOCKED":
    raise SystemExit(1)
if manifest.get("provenance", {}).get("target_data_used_for_basis_or_tuning") is not False:
    raise SystemExit(1)
for p in manifest.get("basis_paths", {}).values():
    if not pathlib.Path(p).exists():
        raise SystemExit(1)
rows = []
for line in jobs_path.read_text(encoding="utf-8").splitlines():
    if not line or line.startswith("#") or line.startswith("job_id\t"):
        continue
    parts = line.split("\t")
    if len(parts) != 2 or not pathlib.Path(parts[1]).exists():
        raise SystemExit(1)
    rows.append(parts)
if len(rows) != 70:
    raise SystemExit(1)
raise SystemExit(0)
PY
}

# Only one process on the shared experiment filesystem may prepare/lock the
# scientific parameters. A second launcher waits here and reuses the result.
exec {LOCK_FD}>"${LOCK_FILE}"
if ! flock -n "${LOCK_FD}"; then
  status "WAITING" "another Defense70 preparation process owns the lock"
  flock "${LOCK_FD}"
fi

if ready_check; then
  status "READY" "locked shadow-only parameters and all 70 formal configs already exist"
  trap - ERR
  exit 0
fi

status "STEP1_REFERENCE" "checking resumable shadow_train seed142 reference trajectory"
REF_INFO="$(python -m src.acm_revision.defense70_prepare write-reference-config \
  --output-config "${REF_CFG}" \
  --output-root "${PREP_ROOT}/reference")"
REF_RUN_DIR="$(printf '%s' "${REF_INFO}" | python -c 'import json,sys; print(json.load(sys.stdin)["run_dir"])')"

REF_OK=1
for round in 1 10 30 50 75 100 125 150; do
  if [[ ! -f "${REF_RUN_DIR}/checkpoints/round_$(printf '%04d' "${round}").pt" ]]; then
    REF_OK=0
    break
  fi
done

if (( REF_OK == 0 )); then
  status "STEP1_REFERENCE" "running/resuming the 150-round shadow-only reference on GPU ${GPU}; this is preparation, not one of the 70 formal runs"
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 \
    python -m src.acm_revision.privacy_capture_runner_resume_v2 \
      --config "${REF_CFG}" --population shadow_train \
      >> "${LOG_FILE}" 2>&1
else
  status "STEP1_REFERENCE" "all required shadow reference checkpoints already exist; reusing them"
fi

for round in 1 10 30 50 75 100 125 150; do
  CKPT="${REF_RUN_DIR}/checkpoints/round_$(printf '%04d' "${round}").pt"
  if [[ ! -f "${CKPT}" ]]; then
    status "FAILED" "reference run finished without required checkpoint ${CKPT}"
    trap - ERR
    exit 3
  fi
done

if [[ ! -f "${MANIFEST}" ]]; then
  status "STEP2_BASIS" "building matched label-composition bases from shadow_train and calibrating rank/alpha on shadow_val only"
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 \
    python -m src.acm_revision.defense70_prepare fit \
      --reference-config "${REF_CFG}" \
      --reference-run-dir "${REF_RUN_DIR}" \
      --output-dir "${PREP_ROOT}" \
      --samples-per-branch "${DEFENSE70_PREP_SAMPLES_PER_BRANCH:-100}" \
      >> "${LOG_FILE}" 2>&1
fi

status "STEP3_GENERATE" "generating the frozen 10 operating points x 7 independent runs"
python -m src.acm_revision.defense70_plan \
  --manifest "${MANIFEST}" \
  --output-dir "${GEN_DIR}" \
  >> "${LOG_FILE}" 2>&1

if ! ready_check; then
  status "FAILED" "post-generation audit failed; formal training was not started"
  trap - ERR
  exit 4
fi

status "READY" "Defense70 preparation complete: 70 formal configs are locked and target data were not used for defense selection"
trap - ERR
echo "Defense70 is READY. jobs.tsv=${JOBS_FILE}"
