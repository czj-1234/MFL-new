#!/usr/bin/env bash
set -uo pipefail

GPU_LIST="${1:-0,1}"
BASE_CONFIG="configs/acm_revision/hateful_memes.yaml"
ROOT="results/acm_revision/diagnostics/local_training_seed42_c07"
CFG_ROOT="configs/acm_revision/generated/local_training_diagnostic"
LOG_ROOT="logs/acm_revision/local_training_diagnostic"
GPU_MIN_FREE_MIB="${GPU_MIN_FREE_MIB:-16000}"
DIAG_ROUNDS="${DIAG_ROUNDS:-5}"

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
if [[ ${#GPUS[@]} -lt 1 || ${#GPUS[@]} -gt 2 ]]; then
  echo "Usage: $0 [gpu] or $0 [gpu0,gpu1]" >&2
  exit 2
fi

mkdir -p "${ROOT}" "${CFG_ROOT}" "${LOG_ROOT}"

# Fast diagnostic only: same strict Target pool, seed42, c=0.7, 20 clients,
# 100 samples/client. We compare local optimization strength without changing data.
python - "${BASE_CONFIG}" "${CFG_ROOT}" "${ROOT}" "${DIAG_ROUNDS}" <<'PY'
import copy, os, sys, yaml
base_path, cfg_root, result_root, rounds = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
with open(base_path, "r", encoding="utf-8") as f:
    base = yaml.safe_load(f)

settings = ["image_only", "text_only", "modality_exclusive", "full_multimodal"]
variants = [
    ("b32_e1", 32, 1),
    ("b16_e1", 16, 1),
    ("b32_e2", 32, 2),
]

for setting in settings:
    for name, batch, epochs in variants:
        c = copy.deepcopy(base)
        c["seed"] = 42
        c["federated"]["num_clients"] = 20
        c["federated"]["samples_per_client"] = 100
        c["federated"]["rounds"] = rounds
        c["federated"]["participation_rate"] = 1.0
        c["federated"]["local_epochs"] = epochs
        c["federated"]["batch_size"] = batch
        c["federated"]["aggregation"] = "fedavg"
        c["federated"]["fedprox_mu"] = 0.0
        c["evaluation"]["eval_every"] = 1
        # 8 workers are excessive for only 100 samples/client and are repeatedly
        # spawned for every local client. Two is a conservative speed-oriented setting.
        c["evaluation"]["num_workers"] = 2
        c["experiment"]["population"] = "target"
        c["experiment"]["setting_name"] = setting
        c["experiment"]["concentration"] = 0.7
        job = f"{setting}__{name}"
        c["experiment"]["job_id"] = job
        c["experiment"]["output_root"] = os.path.join(result_root, job)
        c["update_capture"]["save_all_checkpoints"] = False
        c["update_capture"]["checkpoint_rounds"] = []
        c["update_capture"]["groups"] = []
        path = os.path.join(cfg_root, job + ".yaml")
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(c, f, sort_keys=False, allow_unicode=True)
print("Generated 12 diagnostic configs")
PY

wait_for_gpu_free() {
  local gpu="$1"
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  while true; do
    local free_mib
    free_mib="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -dc '0-9')"
    if [[ "${free_mib}" =~ ^[0-9]+$ ]] && (( free_mib >= GPU_MIN_FREE_MIB )); then
      return 0
    fi
    echo "[$(date '+%F %T')] WAIT GPU=${gpu}: free=${free_mib:-unknown} MiB; need >= ${GPU_MIN_FREE_MIB} MiB"
    sleep 30
  done
}

config_root() {
  python - "$1" <<'PY'
import sys,yaml
with open(sys.argv[1],encoding='utf-8') as f: c=yaml.safe_load(f)
print(c['experiment']['output_root'])
PY
}

job_done() {
  local r
  r="$(config_root "$1")"
  [[ -d "${r}" ]] && find "${r}" -name summary.json -print -quit 2>/dev/null | grep -q .
}

monitor_job() {
  local pid="$1" cfg="$2" gpu="$3" label="$4"
  local root last=-1 total=$((DIAG_ROUNDS * 20))
  root="$(config_root "${cfg}")"
  while kill -0 "${pid}" 2>/dev/null; do
    local n=0
    [[ -d "${root}" ]] && n="$(find "${root}" -type f -path '*/updates/*.npz' 2>/dev/null | wc -l | tr -d ' ')"
    if [[ "${n}" =~ ^[0-9]+$ ]] && (( n > 0 && n != last )); then
      local round client pct
      round=$(( (n - 1) / 20 + 1 ))
      client=$(( (n - 1) % 20 + 1 ))
      pct=$(( n * 100 / total ))
      echo "[$(date '+%F %T')] PROGRESS GPU=${gpu} ${label} round=${round}/${DIAG_ROUNDS} client=${client}/20 (${pct}%)"
      last="${n}"
    fi
    sleep 3
  done
}

run_cfg() {
  local cfg="$1" gpu="$2"
  local job log root pid mon status=0
  job="$(basename "${cfg}" .yaml)"
  log="${LOG_ROOT}/${job}.log"
  root="$(config_root "${cfg}")"

  if job_done "${cfg}"; then
    echo "[$(date '+%F %T')] SKIP GPU=${gpu} ${job} already complete"
    return 0
  fi
  [[ -d "${root}" ]] && rm -rf "${root}"
  wait_for_gpu_free "${gpu}"

  echo "[$(date '+%F %T')] START GPU=${gpu} ${job}"
  PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${gpu}" \
    python -m src.acm_revision.cli run-fl --config "${cfg}" > "${log}" 2>&1 &
  pid=$!
  monitor_job "${pid}" "${cfg}" "${gpu}" "${job}" &
  mon=$!
  wait "${pid}" || status=$?
  wait "${mon}" 2>/dev/null || true

  if (( status == 0 )) && job_done "${cfg}"; then
    echo "[$(date '+%F %T')] DONE GPU=${gpu} ${job}"
    find "${root}" -type f -name best_model.pt -delete 2>/dev/null || true
    find "${root}" -type d -name updates -prune -exec rm -rf {} + 2>/dev/null || true
    return 0
  fi
  echo "[$(date '+%F %T')] FAIL GPU=${gpu} ${job}" >&2
  tail -n 80 "${log}" >&2 || true
  return 1
}

worker() {
  local gpu="$1" group="$2"
  local settings
  if [[ "${group}" == "A" ]]; then
    settings=(image_only modality_exclusive)
  else
    settings=(text_only full_multimodal)
  fi
  local setting variant cfg
  for setting in "${settings[@]}"; do
    for variant in b32_e1 b16_e1 b32_e2; do
      cfg="${CFG_ROOT}/${setting}__${variant}.yaml"
      run_cfg "${cfg}" "${gpu}" || return 1
    done
  done
}

if [[ ${#GPUS[@]} -eq 2 ]]; then
  worker "${GPUS[0]}" A & p0=$!
  worker "${GPUS[1]}" B & p1=$!
  failures=0
  wait "${p0}" || failures=$((failures+1))
  wait "${p1}" || failures=$((failures+1))
  (( failures == 0 )) || exit 1
else
  worker "${GPUS[0]}" A
  worker "${GPUS[0]}" B
fi

python - "${ROOT}" <<'PY'
import csv, json, os, sys
from pathlib import Path
root = Path(sys.argv[1])
rows=[]
for job_dir in sorted(root.iterdir()):
    if not job_dir.is_dir(): continue
    cfgs=list(job_dir.rglob('resolved_config.json'))
    mets=list(job_dir.rglob('round_metrics.csv'))
    sums=list(job_dir.rglob('summary.json'))
    if not (cfgs and mets and sums): continue
    with open(cfgs[0],encoding='utf-8') as f: c=json.load(f)
    with open(mets[0],encoding='utf-8') as f:
        data=list(csv.DictReader(f))
    last=data[-1]
    with open(sums[0],encoding='utf-8') as f: s=json.load(f)
    rows.append({
      'setting': c['experiment']['setting_name'],
      'batch_size': c['federated']['batch_size'],
      'local_epochs': c['federated']['local_epochs'],
      'rounds': c['federated']['rounds'],
      'final_val_acc': last.get('val_acc'),
      'final_val_macro_f1': last.get('val_macro_f1'),
      'final_val_auroc': last.get('val_auroc'),
      'final_test_acc': last.get('test_acc'),
      'final_test_macro_f1': last.get('test_macro_f1'),
      'final_test_auroc': last.get('test_auroc'),
      'runtime_seconds': s.get('runtime_seconds'),
    })
out=root/'diagnostic_summary.csv'
with open(out,'w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['setting'])
    w.writeheader(); w.writerows(rows)
print(f"[OK] Diagnostic summary: {out}")
for r in rows:
    print(r)
PY
