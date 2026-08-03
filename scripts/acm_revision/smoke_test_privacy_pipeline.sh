#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
SEED=999
CONCENTRATION=0.7
ROUNDS=2
JOB_SCRIPT="scripts/acm_revision/run_privacy_capture_job.sh"
ROOT="results/acm_revision/privacy_capture_r${ROUNDS}"
OUT="results/acm_revision/privacy_smoke_postprocess"
PASS_MARKER="${OUT}/c0p7_r2/SMOKE_PASS"

if [[ -f "${PASS_MARKER}" ]]; then
  echo "[SMOKE SKIP] Existing PASS marker: ${PASS_MARKER}"
  exit 0
fi

export PRIVACY_NUM_CLIENTS=4
export PRIVACY_SAMPLES_PER_CLIENT=8
export PRIVACY_MAX_LOCAL_STEPS=1
export PRIVACY_SKETCH_DIM=256
export GPU_MIN_FREE_MIB="${GPU_MIN_FREE_MIB:-12000}"

for population in shadow_train shadow_val target; do
  bash "${JOB_SCRIPT}" "${GPU}" "${CONCENTRATION}" "${ROUNDS}" "${SEED}" "${population}"
done

python -m src.acm_revision.privacy_postprocess \
  --root "${ROOT}" \
  --concentration "${CONCENTRATION}" \
  --rounds "${ROUNDS}" \
  --seeds "${SEED}" \
  --output-root "${OUT}" \
  --smoke

REPORT="${OUT}/c0p7_r2/postprocess_report.json"
python - "${REPORT}" "${PASS_MARKER}" <<'PY'
import json
import pathlib
import sys
report_path, marker_path = map(pathlib.Path, sys.argv[1:])
report = json.loads(report_path.read_text(encoding="utf-8"))
if report.get("status") != "PASS":
    raise SystemExit(f"Smoke postprocess did not pass: {report}")
marker_path.parent.mkdir(parents=True, exist_ok=True)
marker_path.write_text("PASS\n", encoding="utf-8")
print(f"[SMOKE PASS] {marker_path}")
PY
