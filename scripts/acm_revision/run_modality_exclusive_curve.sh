#!/usr/bin/env bash
set -uo pipefail

GPU="${1:-1}"
CONCENTRATION="${2:-0.5}"
ROUNDS="${3:-50}"
BASE_CONFIG="configs/acm_revision/hateful_memes.yaml"
GPU_MIN_FREE_MIB="${GPU_MIN_FREE_MIB:-16000}"

# Keep diagnostic outputs separate from formal E1 results.
TAG="c${CONCENTRATION//./p}_r${ROUNDS}"
ROOT="results/acm_revision/diagnostics/modality_exclusive_curve_seed42_${TAG}"
CFG_ROOT="configs/acm_revision/generated/modality_exclusive_curve"
LOG_ROOT="logs/acm_revision/modality_exclusive_curve"
CFG="${CFG_ROOT}/modality_exclusive_seed42_${TAG}.yaml"
LOG="${LOG_ROOT}/modality_exclusive_seed42_${TAG}.log"

if [[ "${GPU}" == *","* ]]; then
  echo "Usage: $0 [single-gpu-id] [concentration] [rounds]" >&2
  echo "Example: $0 1 0.5 50" >&2
  exit 2
fi

mkdir -p "${ROOT}" "${CFG_ROOT}" "${LOG_ROOT}"

python - "${BASE_CONFIG}" "${CFG}" "${ROOT}" "${CONCENTRATION}" "${ROUNDS}" <<'PY'
import copy
import sys
import yaml

base_path, cfg_path, result_root, concentration, rounds = sys.argv[1:]
concentration = float(concentration)
rounds = int(rounds)

if not (0.5 <= concentration <= 1.0):
    raise SystemExit("For this binary Hateful Memes diagnostic, concentration must be in [0.5, 1.0].")
if rounds <= 0:
    raise SystemExit("rounds must be positive")

with open(base_path, "r", encoding="utf-8") as f:
    cfg = copy.deepcopy(yaml.safe_load(f))

cfg["seed"] = 42
cfg["federated"]["num_clients"] = 20
cfg["federated"]["partition_mode"] = "fixed"
cfg["federated"]["samples_per_client"] = 100
cfg["federated"]["rounds"] = rounds
cfg["federated"]["participation_rate"] = 1.0
cfg["federated"]["local_epochs"] = 1
# Chosen from the local-training diagnostic: similar runtime to batch 32,
# but slightly better utility and more local optimization steps.
cfg["federated"]["batch_size"] = 16
cfg["federated"]["aggregation"] = "fedavg"
cfg["federated"]["fedprox_mu"] = 0.0

# Record a useful learning curve without evaluating every round.
cfg["evaluation"]["eval_every"] = 5
cfg["evaluation"]["num_workers"] = 2

cfg["experiment"]["population"] = "target"
cfg["experiment"]["setting_name"] = "modality_exclusive"
cfg["experiment"]["concentration"] = concentration
cfg["experiment"]["job_id"] = f"modality_exclusive_seed42_c{concentration}_r{rounds}"
cfg["experiment"]["output_root"] = result_root

# Utility/convergence diagnostic only. Keep lightweight progress marker files,
# but do not retain model checkpoints or parameter payload groups.
cfg["update_capture"]["save_all_checkpoints"] = False
cfg["update_capture"]["checkpoint_rounds"] = []
cfg["update_capture"]["groups"] = []

with open(cfg_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

print(
    f"Generated: setting=modality_exclusive seed=42 concentration={concentration} "
    f"rounds={rounds} clients=20 samples/client=100 batch=16 local_epochs=1"
)
PY

config_output_root() {
  python - "$1" <<'PY'
import sys, yaml
with open(sys.argv[1], encoding='utf-8') as f:
    c = yaml.safe_load(f)
print(c['experiment']['output_root'])
PY
}

job_done() {
  local root
  root="$(config_output_root "$1")"
  [[ -d "${root}" ]] && find "${root}" -name summary.json -print -quit 2>/dev/null | grep -q .
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

monitor_job() {
  local pid="$1"
  local root last_count=-1 last_eval=-1 total_updates=$((ROUNDS * 20))
  root="$(config_output_root "${CFG}")"

  while kill -0 "${pid}" 2>/dev/null; do
    local count=0
    [[ -d "${root}" ]] && count="$(find "${root}" -type f -path '*/updates/*.npz' 2>/dev/null | wc -l | tr -d ' ')"

    if [[ "${count}" =~ ^[0-9]+$ ]] && (( count > 0 && count != last_count )); then
      local round client pct
      round=$(( (count - 1) / 20 + 1 ))
      client=$(( (count - 1) % 20 + 1 ))
      pct=$(( count * 100 / total_updates ))
      echo "[$(date '+%F %T')] PROGRESS GPU=${GPU} c=${CONCENTRATION} round=${round}/${ROUNDS} client=${client}/20 (${pct}%)"
      last_count="${count}"
    fi

    local metrics
    metrics="$(find "${root}" -name round_metrics.csv -print -quit 2>/dev/null || true)"
    if [[ -n "${metrics}" && -s "${metrics}" ]]; then
      local eval_round
      eval_round="$(awk -F, 'NR>1 {r=$2} END {if (r!="") print r}' "${metrics}")"
      if [[ "${eval_round}" =~ ^[0-9]+$ ]] && (( eval_round != last_eval )); then
        awk -F, -v gpu="${GPU}" -v conc="${CONCENTRATION}" '
          NR>1 {r=$2; va=$4; vf=$5; vu=$9; ta=$11; tf=$12; tu=$16}
          END {
            if (r != "") {
              cmd="date +\"%F %T\""; cmd | getline now; close(cmd);
              printf("[%s] METRICS GPU=%s c=%s round=%s val_acc=%s val_f1=%s val_auroc=%s test_acc=%s test_f1=%s test_auroc=%s\n", now,gpu,conc,r,va,vf,vu,ta,tf,tu);
              fflush();
            }
          }
        ' "${metrics}"
        last_eval="${eval_round}"
      fi
    fi
    sleep 3
  done
}

ROOT_FROM_CFG="$(config_output_root "${CFG}")"
if job_done "${CFG}"; then
  echo "[SKIP] This curve is already complete: ${ROOT_FROM_CFG}"
  exit 0
fi

# A partial old attempt should never contaminate a restart.
if [[ -d "${ROOT_FROM_CFG}" ]]; then
  rm -rf "${ROOT_FROM_CFG}"
fi

wait_for_gpu_free

echo "============================================================"
echo "START modality_exclusive convergence curve"
echo "GPU=${GPU} seed=42 concentration=${CONCENTRATION} rounds=${ROUNDS}"
echo "20 clients x 100 samples, batch=16, local_epochs=1, FedAvg"
echo "Evaluation: round 1, every 5 rounds, and final round"
echo "Log: ${LOG}"
echo "============================================================"

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${GPU}" \
  python -m src.acm_revision.cli run-fl --config "${CFG}" > "${LOG}" 2>&1 &
TRAIN_PID=$!

cleanup() {
  kill -TERM "${TRAIN_PID}" 2>/dev/null || true
}
trap cleanup INT TERM

monitor_job "${TRAIN_PID}" &
MON_PID=$!

STATUS=0
wait "${TRAIN_PID}" || STATUS=$?
wait "${MON_PID}" 2>/dev/null || true
trap - INT TERM

if (( STATUS != 0 )) || ! job_done "${CFG}"; then
  echo "[FAIL] Training failed. Last 100 log lines:" >&2
  tail -n 100 "${LOG}" >&2 || true
  exit 1
fi

# Keep summary/config/metrics/manifests; remove only transient progress markers.
find "${ROOT_FROM_CFG}" -type d -name updates -prune -exec rm -rf {} + 2>/dev/null || true
find "${ROOT_FROM_CFG}" -type f -name best_model.pt -delete 2>/dev/null || true

echo "============================================================"
echo "[DONE] modality_exclusive c=${CONCENTRATION}, ${ROUNDS} rounds"
echo "Result root: ${ROOT_FROM_CFG}"
echo "Check round_metrics.csv for rounds 1,5,10,...,${ROUNDS}."
echo "============================================================"
