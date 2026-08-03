#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?GPU id required}"
SETTING="${2:?setting required}"
CONCENTRATION="${3:?concentration required}"
POPULATION="${4:?population required}"
SEED="${5:?seed required}"
ROUNDS="${6:-150}"

BASE_CONFIG="configs/acm_revision/hateful_memes.yaml"
GPU_MIN_FREE_MIB="${GPU_MIN_FREE_MIB:-16000}"
SKETCH_DIM="${CORE72_SKETCH_DIM:-16384}"
NUM_CLIENTS="${CORE72_NUM_CLIENTS:-12}"
MAX_LOCAL_STEPS="${CORE72_MAX_LOCAL_STEPS:-none}"
RESULT_BASE="${CORE72_RESULT_ROOT:-results/acm_revision/core72_r${ROUNDS}}"
CFG_BASE="${CORE72_CONFIG_ROOT:-configs/acm_revision/generated/core72_r${ROUNDS}}"
LOG_BASE="${CORE72_LOG_ROOT:-logs/acm_revision/core72_r${ROUNDS}}"

case "${SETTING}" in
  image_only|text_only|modality_exclusive) ;;
  *) echo "Invalid setting: ${SETTING}" >&2; exit 2 ;;
esac
case "${POPULATION}" in
  shadow_train|shadow_val|target) ;;
  *) echo "Invalid population: ${POPULATION}" >&2; exit 2 ;;
esac
case "${POPULATION}" in
  shadow_train|target) DEFAULT_SAMPLES=200 ;;
  shadow_val) DEFAULT_SAMPLES=100 ;;
esac
SAMPLES_PER_CLIENT="${CORE72_SAMPLES_PER_CLIENT:-${DEFAULT_SAMPLES}}"

if [[ "${GPU}" == *","* ]]; then
  echo "A single GPU id is required, not a comma-separated list." >&2
  exit 2
fi

CTAG="${CONCENTRATION//./p}"
JOB_TAG="${SETTING}__c${CTAG}__${POPULATION}__seed${SEED}__n${NUM_CLIENTS}__s${SAMPLES_PER_CLIENT}__r${ROUNDS}"
ROOT="${RESULT_BASE}/${JOB_TAG}"
CFG="${CFG_BASE}/${JOB_TAG}.yaml"
LOG="${LOG_BASE}/${JOB_TAG}.log"
mkdir -p "${ROOT}" "$(dirname "${CFG}")" "$(dirname "${LOG}")"

python - "${BASE_CONFIG}" "${CFG}" "${ROOT}" "${SETTING}" "${CONCENTRATION}" "${ROUNDS}" "${SEED}" "${POPULATION}" "${SAMPLES_PER_CLIENT}" "${SKETCH_DIM}" "${NUM_CLIENTS}" "${MAX_LOCAL_STEPS}" <<'PY'
import copy
import sys
import yaml

(
    base_path, cfg_path, output_root, setting, concentration, rounds, seed,
    population, samples_per_client, sketch_dim, num_clients, max_local_steps,
) = sys.argv[1:]
concentration = float(concentration)
rounds = int(rounds)
seed = int(seed)
samples_per_client = int(samples_per_client)
sketch_dim = int(sketch_dim)
num_clients = int(num_clients)
max_local_steps = None if max_local_steps.lower() in {"none", "null", ""} else int(max_local_steps)

if setting not in {"image_only", "text_only", "modality_exclusive"}:
    raise SystemExit(f"Invalid setting: {setting}")
if population not in {"shadow_train", "shadow_val", "target"}:
    raise SystemExit(f"Invalid population: {population}")
if concentration not in {0.5, 0.7, 0.9}:
    raise SystemExit("Core72 concentration must be one of 0.5, 0.7, 0.9")
if rounds <= 0 or samples_per_client <= 0 or sketch_dim <= 0 or num_clients <= 0:
    raise SystemExit("rounds, samples_per_client, sketch_dim and num_clients must be positive")
if setting == "modality_exclusive" and num_clients % 4 != 0:
    raise SystemExit("modality_exclusive requires num_clients divisible by 4")

with open(base_path, "r", encoding="utf-8") as f:
    cfg = copy.deepcopy(yaml.safe_load(f))

cfg["seed"] = seed
fed = cfg["federated"]
fed["num_clients"] = num_clients
fed["partition_mode"] = "fixed"
fed["samples_per_client"] = samples_per_client
fed.pop("client_sample_counts", None)
fed["rounds"] = rounds
fed["participation_rate"] = 1.0
fed["min_participants"] = num_clients
fed["local_epochs"] = 1
fed["batch_size"] = 16
fed["max_local_steps"] = max_local_steps
fed["aggregation"] = "fedavg"
fed["fedprox_mu"] = 0.0

cfg["evaluation"]["eval_every"] = 5
cfg["evaluation"]["num_workers"] = 2

exp = cfg["experiment"]
exp["population"] = population
exp["setting_name"] = setting
exp["concentration"] = concentration
exp["balanced_attribution_control"] = bool(concentration == 0.5)
exp["job_id"] = f"core72_{setting}_c{concentration}_{population}_seed{seed}_n{num_clients}_s{samples_per_client}_r{rounds}"
exp["output_root"] = output_root

milestones = [x for x in [1, 5, 10, 20, 30, 50, 75, 100, 125, 150] if x <= rounds]
if rounds not in milestones:
    milestones.append(rounds)
milestones = sorted(set(milestones))

cfg["privacy_capture"] = {
    "every_round_exact_groups": ["classifier_bias", "classifier_weight", "classifier_head"],
    "milestone_exact_groups": ["fusion", "missing_modality"],
    "milestone_sketch_groups": ["image_encoder", "text_encoder", "all_shared", "full_update"],
    "full_group_projection_groups": ["full_update"],
    "full_group_projection_dim": 2048,
    "full_group_projection_seed": 20260804,
    "milestone_rounds": milestones,
    "sketch_dim": sketch_dim,
    "sketch_seed": 20260803,
    "representation_scope_note": (
        "Encoder and all_shared groups use deterministic coordinate sketches. "
        "The full_update group uses a deterministic signed feature-hash projection in which every coordinate contributes."
    ),
}

cfg.setdefault("update_capture", {})["save_all_checkpoints"] = False
reference_checkpoint_job = (
    setting == "modality_exclusive"
    and abs(concentration - 0.7) < 1e-12
    and population == "target"
    and seed in {42, 43, 44}
)
reference_rounds = [x for x in [1, 10, 30, 50, 75, 100, 125, 150] if x <= rounds]
cfg["update_capture"]["checkpoint_rounds"] = reference_rounds if reference_checkpoint_job else []
cfg["update_capture"]["model_checkpoint_rounds"] = reference_rounds if reference_checkpoint_job else []
cfg["update_capture"]["retain_for_identical_checkpoint_contrast"] = reference_checkpoint_job
cfg["update_capture"]["storage_dtype"] = "float16"

with open(cfg_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

print(
    f"Generated Core72 job: setting={setting} c={concentration} population={population} "
    f"seed={seed} rounds={rounds} clients={num_clients} samples/client={samples_per_client}"
)
PY

job_done() {
  python - "${ROOT}" "${SETTING}" "${CONCENTRATION}" "${ROUNDS}" "${SEED}" "${POPULATION}" "${NUM_CLIENTS}" <<'PY'
import json, math, pathlib, sys
root, setting, concentration, rounds, seed, population, num_clients = sys.argv[1:]
concentration, rounds, seed, num_clients = float(concentration), int(rounds), int(seed), int(num_clients)
for summary_path in pathlib.Path(root).rglob("summary.json"):
    run_dir = summary_path.parent
    manifest_path, metadata_path = run_dir / "capture_manifest.json", run_dir / "update_metadata.csv"
    if not manifest_path.exists() or not metadata_path.exists():
        continue
    try:
        summary = json.loads(summary_path.read_text())
        manifest = json.loads(manifest_path.read_text())
    except Exception:
        continue
    if (
        int(summary.get("seed", -1)) == seed
        and int(summary.get("rounds", -1)) == rounds
        and int(summary.get("num_clients", -1)) == num_clients
        and str(summary.get("setting_name")) == setting
        and str(summary.get("population")) == population
        and math.isclose(float(summary.get("concentration")), concentration, abs_tol=1e-9)
        and manifest.get("status") == "PASS"
        and metadata_path.stat().st_size > 0
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

echo "START CORE72: GPU=${GPU} setting=${SETTING} c=${CONCENTRATION} population=${POPULATION} seed=${SEED} rounds=${ROUNDS}"
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${GPU}" \
  python -m src.acm_revision.privacy_capture_runner_v2 \
    --config "${CFG}" --population "${POPULATION}" > "${LOG}" 2>&1 &
PID=$!
cleanup() { kill -TERM "${PID}" 2>/dev/null || true; }
trap cleanup INT TERM
STATUS=0
wait "${PID}" || STATUS=$?
trap - INT TERM
if (( STATUS != 0 )) || ! job_done; then
  echo "[FAIL] ${JOB_TAG}. Last 200 log lines:" >&2
  tail -n 200 "${LOG}" >&2 || true
  exit 1
fi

find "${ROOT}" -type f -name best_model.pt -delete 2>/dev/null || true
if [[ "${SETTING}" == "modality_exclusive" && "${CONCENTRATION}" == "0.7" && "${POPULATION}" == "target" && "${SEED}" =~ ^(42|43|44)$ ]]; then
  echo "[KEEP CHECKPOINTS] identical-checkpoint contrast reference trajectory"
else
  find "${ROOT}" -type d -name checkpoints -prune -exec rm -rf {} + 2>/dev/null || true
fi

echo "[DONE PASS] ${JOB_TAG}"
