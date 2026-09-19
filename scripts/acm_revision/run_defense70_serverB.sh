#!/usr/bin/env bash
set -euo pipefail

export DEFENSE70_MIN_FREE_MIB="${DEFENSE70_MIN_FREE_MIB:-23000}"
export DEFENSE70_STARTUP_WAIT="${DEFENSE70_STARTUP_WAIT:-90}"
export DEFENSE70_MAX_RETRIES="${DEFENSE70_MAX_RETRIES:-2}"
export DEFENSE70_RETRY_SLEEP="${DEFENSE70_RETRY_SLEEP:-60}"
export DEFENSE70_CHECKPOINT_ROOT="${DEFENSE70_CHECKPOINT_ROOT:-checkpoints/acm_revision/defense70_r150}"
export DEFENSE70_CHECKPOINT_EVERY="${DEFENSE70_CHECKPOINT_EVERY:-15}"
export DEFENSE70_CHECKPOINT_KEEP_LAST="${DEFENSE70_CHECKPOINT_KEEP_LAST:-2}"
export MFL_NFS_WRITE_RETRIES="${MFL_NFS_WRITE_RETRIES:-8}"

JOBS_FILE="${DEFENSE70_JOBS_FILE:-configs/acm_revision/generated/defense70/jobs.tsv}"
MANIFEST="${DEFENSE70_PREP_ROOT:-results/acm_revision/defense70_prep}/locked_params.yaml"

# Server A is the single preparation owner. Server B must use exactly the same
# frozen basis/parameters, so it waits rather than independently re-fitting
# anything. On the normal shared experiment filesystem this becomes READY as
# soon as Server A finishes preparation.
while [[ ! -f "${JOBS_FILE}" || ! -f "${MANIFEST}" ]]; do
  echo "[$(date '+%F %T')] Server B waiting for Server A Defense70 preparation: ${JOBS_FILE} / ${MANIFEST}"
  sleep 30
done

python - "${MANIFEST}" "${JOBS_FILE}" <<'PY'
import pathlib, sys, yaml
manifest_path, jobs_path = map(pathlib.Path, sys.argv[1:])
manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
if manifest.get("status") != "LOCKED":
    raise SystemExit("Defense70 manifest is not LOCKED")
if manifest.get("provenance", {}).get("target_data_used_for_basis_or_tuning") is not False:
    raise SystemExit("Defense70 manifest does not certify target-data isolation")
for path in manifest.get("basis_paths", {}).values():
    if not pathlib.Path(path).exists():
        raise SystemExit(f"Missing shared Defense70 basis: {path}")
rows = [
    line for line in jobs_path.read_text(encoding="utf-8").splitlines()
    if line and not line.startswith("#") and not line.startswith("job_id\t")
]
if len(rows) != 70:
    raise SystemExit(f"Expected 70 Defense70 jobs, found {len(rows)}")
PY

exec bash scripts/acm_revision/run_defense70_server.sh B "${1:-0}" "${2:-1}"
