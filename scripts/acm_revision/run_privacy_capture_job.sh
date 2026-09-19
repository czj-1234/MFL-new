#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
CONCENTRATION="${2:-0.7}"
ROUNDS="${3:-150}"
SEED="${4:-42}"
POPULATION="${5:-target}"
BASE_CONFIG="configs/acm_revision/hateful_memes.yaml"
GPU_MIN_FREE_MIB="${GPU_MIN_FREE_MIB:-16000}"
SKETCH_DIM="${PRIVACY_SKETCH_DIM:-16384}"
NUM_CLIENTS="${PRIVACY_NUM_CLIENTS:-12}"
MAX_LOCAL_STEPS="${PRIVACY_MAX_LOCAL_STEPS:-none}"

case "${POPULATION}" in
  shadow_train|target) SAMPLES_PER_CLIENT=200 ;;
  shadow_val) SAMPLES_PER_CLIENT=100 ;;
  *) echo "Population must be shadow_train, shadow_val, or target." >&2; exit 2 ;;
esac
SAMPLES_PER_CLIENT="${PRIVACY_SAMPLES_PER_CLIENT:-${SAMPLES_PER_CLIENT}}"

if [[ "${GPU}" == *","* ]]; then
  echo "Usage: $0 [gpu] [concentration] [rounds] [seed] [population]" >&2
  exit 2
fi

CTAG="${CONCENTRATION//./p}"
JOB_TAG="seed${SEED}_c${CTAG}_r${ROUNDS}_${POPULATION}"
ROOT="results/acm_revision/privacy_capture_r${ROUNDS}/${JOB_TAG}"
CFG_ROOT="configs/acm_revision/generated/privacy_capture_r${ROUNDS}"
LOG_ROOT="logs/acm_revision/privacy_capture_r${ROUNDS}"
CFG="${CFG_ROOT}/${JOB_TAG}.yaml"
LOG="${LOG_ROOT}/${JOB_TAG}.log"
mkdir -p "${ROOT}" "${CFG_ROOT}" "${LOG_ROOT}"

python - "${BASE_CONFIG}" "${CFG}" "${ROOT}" "${CONCENTRATION}" "${ROUNDS}" "${SEED}" "${POPULATION}" "${SAMPLES_PER_CLIENT}" "${SKETCH_DIM}" "${NUM_CLIENTS}" "${MAX_LOCAL_STEPS}" <<'PY'
import copy
import sys
import yaml

(
    base_path,
    cfg_path,
    output_root,
    concentration,
    rounds,
    seed,
    population,
    samples_per_client,
    sketch_dim,
    num_clients,
    max_local_steps,
) = sys.argv[1:]
concentration = float(concentration)
rounds = int(rounds)
seed = int(seed)
samples_per_client = int(samples_per_client)
sketch_dim = int(sketch_dim)
num_clients = int(num_clients)
max_local_steps = None if str(max_local_steps).lower() in {"none", "null", ""} else int(max_local_steps)

if not 0.5 <= concentration <= 1.0:
    raise SystemExit("Concentration must be in [0.5, 1.0].")
if rounds <= 0 or sketch_dim <= 0:
    raise SystemExit("rounds and sketch_dim must be positive")

with open(base_path, "r", encoding="utf-8") as f:
    cfg = copy.deepcopy(yaml.safe_load(f))

cfg["seed"] = seed
cfg["federated"]["num_clients"] = num_clients
cfg["federated"]["partition_mode"] = "fixed"
cfg["federated"]["samples_per_client"] = samples_per_client
cfg["federated"].pop("client_sample_counts", None)
cfg["federated"]["rounds"] = rounds
cfg["federated"]["participation_rate"] = 1.0
cfg["federated"]["local_epochs"] = 1
cfg["federated"]["batch_size"] = 16
cfg["federated"]["max_local_steps"] = max_local_steps
cfg["federated"]["aggregation"] = "fedavg"
cfg["federated"]["fedprox_mu"] = 0.0

cfg["evaluation"]["eval_every"] = 5
cfg["evaluation"]["num_workers"] = 2

cfg["experiment"]["population"] = population
cfg["experiment"]["setting_name"] = "modality_exclusive"
cfg["experiment"]["concentration"] = concentration
cfg["experiment"]["job_id"] = (
    f"privacy_capture_seed{seed}_c{concentration}_r{rounds}_{population}"
)
cfg["experiment"]["output_root"] = output_root

milestones = [x for x in [1, 5, 10, 20, 30, 50, 75, 100, 125, 150] if x <= rounds]
if rounds not in milestones:
    milestones.append(rounds)
milestones = sorted(set(milestones))

cfg["privacy_capture"] = {
    "every_round_exact_groups": [
        "classifier_bias",
        "classifier_weight",
        "classifier_head",
    ],
    "milestone_exact_groups": ["fusion", "missing_modality"],
    "milestone_sketch_groups": [
        "image_encoder",
        "text_encoder",
        "all_shared",
        "full_update",
    ],
    "milestone_rounds": milestones,
    "sketch_dim": sketch_dim,
    "sketch_seed": 20260803,
}

cfg["update_capture"]["save_all_checkpoints"] = False
cfg["update_capture"]["checkpoint_rounds"] = []
cfg["update_capture"]["model_checkpoint_rounds"] = []
cfg["update_capture"]["storage_dtype"] = "float16"

with open(cfg_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

print(
    f"Generated privacy job: seed={seed} c={concentration} population={population} "
    f"rounds={rounds} clients={num_clients} samples/client={samples_per_client} "
    f"max_local_steps={max_local_steps} sketch_dim={sketch_dim}"
)
PY

job_done() {
  python - "${ROOT}" "${SEED}" "${CONCENTRATION}" "${ROUNDS}" "${POPULATION}" <<'PY'
import json
import math
import pathlib
import sys
root, seed, concentration, rounds, population = sys.argv[1:]
seed, rounds = int(seed), int(rounds)
concentration = float(concentration)
for summary_path in pathlib.Path(root).rglob("summary.json"):
    manifest_path = summary_path.parent / "capture_manifest.json"
    if not manifest_path.exists():
        continue
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        continue
    if (
        int(summary.get("seed", -1)) == seed
        and int(summary.get("rounds", -1)) == rounds
        and str(summary.get("population")) == population
        and math.isclose(float(summary.get("concentration")), concentration, abs_tol=1e-9)
        and manifest.get("status") == "PASS"
    ):
        raise SystemExit(0)
raise SystemExit(1)
PY
}

wait_for_gpu_free() {
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  while true; do
    local free_mib
    free_mib="$(nvidia-smi -i "${GPU}" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -dc '0-9')"
    if [[ "${free_mib}" =~ ^[0-9]+$ ]] && (( free_mib >= GPU_MIN_FREE_MIB )); then
      echo "[$(date '+%F %T')] GPU=${GPU} ready: free=${free_mib} MiB"
      return 0
    fi
    echo "[$(date '+%F %T')] WAIT GPU=${GPU}: free=${free_mib:-unknown} MiB; need >= ${GPU_MIN_FREE_MIB} MiB"
    sleep 30
  done
}

if job_done; then
  echo "[SKIP PASS] ${JOB_TAG}"
  exit 0
fi

rm -rf "${ROOT}"
mkdir -p "${ROOT}"
wait_for_gpu_free

echo "============================================================"
echo "START COMPLETE PRIVACY CAPTURE"
echo "GPU=${GPU} seed=${SEED} c=${CONCENTRATION} population=${POPULATION} rounds=${ROUNDS}"
echo "${NUM_CLIENTS} clients, samples/client=${SAMPLES_PER_CLIENT}, batch=16, local_epochs=1, max_steps=${MAX_LOCAL_STEPS}"
echo "Exact head updates every round; exact fusion + sketched encoder/full updates at milestones"
echo "Log: ${LOG}"
echo "============================================================"

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${GPU}" \
  python -m src.acm_revision.privacy_capture_runner \
    --config "${CFG}" --population "${POPULATION}" > "${LOG}" 2>&1 &
PID=$!

cleanup() {
  kill -TERM "${PID}" 2>/dev/null || true
}
trap cleanup INT TERM

STATUS=0
wait "${PID}" || STATUS=$?
trap - INT TERM

if (( STATUS != 0 )) || ! job_done; then
  echo "[FAIL] ${JOB_TAG}. Last 150 log lines:" >&2
  tail -n 150 "${LOG}" >&2 || true
  exit 1
fi

find "${ROOT}" -type f -name best_model.pt -delete 2>/dev/null || true
find "${ROOT}" -type d -name checkpoints -prune -exec rm -rf {} + 2>/dev/null || true

echo "[DONE PASS] ${JOB_TAG}"
