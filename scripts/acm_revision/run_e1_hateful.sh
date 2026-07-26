#!/usr/bin/env bash
set -uo pipefail

GPU_LIST="${1:-0}"
MODE="${2:-all}"   # all | smoke
MATRIX="configs/acm_revision/e1_hateful.yaml"
PROFILE="E1_task_performance_hateful"
GENERATED_ROOT="configs/acm_revision/generated"
LOG_ROOT="logs/acm_revision/${PROFILE}"
RESULT_ROOT="results/acm_revision/${PROFILE}"

if [[ "${MODE}" != "all" && "${MODE}" != "smoke" ]]; then
  echo "Usage: $0 [comma-separated-gpu-ids] [all|smoke]" >&2
  exit 2
fi

python -m src.acm_revision.cli generate-jobs \
  --matrix "${MATRIX}" \
  --profile "${PROFILE}" \
  --output-dir "${GENERATED_ROOT}"

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
if [[ ${#GPUS[@]} -eq 0 ]]; then
  echo "[ERROR] No GPUs supplied." >&2
  exit 2
fi

mapfile -t CONFIGS < <(find "${GENERATED_ROOT}/${PROFILE}" -maxdepth 1 -name '*.yaml' | sort)
mkdir -p "${LOG_ROOT}" "${RESULT_ROOT}"

if [[ ${#CONFIGS[@]} -ne 200 ]]; then
  echo "[ERROR] Expected exactly 200 Hateful E1 jobs, found ${#CONFIGS[@]}." >&2
  exit 1
fi

config_seed() {
  local config="$1"
  awk '/^seed:[[:space:]]*/ {print $2; exit}' "${config}"
}

config_output_root() {
  local config="$1"
  python - "${config}" <<'PY'
import sys, yaml
with open(sys.argv[1], "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
print(cfg["experiment"]["output_root"])
PY
}

config_meta() {
  local config="$1"
  python - "${config}" <<'PY'
import sys, yaml
with open(sys.argv[1], "r", encoding="utf-8") as f:
    c = yaml.safe_load(f)
print(
    f"seed={c['seed']} "
    f"setting={c['experiment']['setting_name']} "
    f"c={c['experiment']['concentration']} "
    f"rounds={c['federated']['rounds']} "
    f"clients={c['federated']['num_clients']}"
)
PY
}

config_rounds() {
  local config="$1"
  python - "${config}" <<'PY'
import sys, yaml
with open(sys.argv[1], "r", encoding="utf-8") as f:
    c = yaml.safe_load(f)
print(int(c["federated"]["rounds"]))
PY
}

config_clients() {
  local config="$1"
  python - "${config}" <<'PY'
import sys, yaml
with open(sys.argv[1], "r", encoding="utf-8") as f:
    c = yaml.safe_load(f)
print(int(c["federated"]["num_clients"]))
PY
}

job_done() {
  local config="$1"
  local root
  root="$(config_output_root "${config}")"
  [[ -d "${root}" ]] && find "${root}" -name summary.json -print -quit 2>/dev/null | grep -q .
}

cleanup_heavy_artifacts() {
  local config="$1"
  local root
  root="$(config_output_root "${config}")"
  [[ -d "${root}" ]] || return 0
  find "${root}" -type f -name 'best_model.pt' -delete 2>/dev/null || true
  find "${root}" -type d -name checkpoints -prune -exec rm -rf {} + 2>/dev/null || true
  # E1 groups are empty. The zero-payload update files are used only as live
  # client-progress markers and are removed after the job completes.
  find "${root}" -type d -name updates -prune -exec rm -rf {} + 2>/dev/null || true
}

progress_monitor() {
  local pid="$1"
  local config="$2"
  local gpu="$3"
  local job_label="$4"
  local root rounds clients total_updates
  root="$(config_output_root "${config}")"
  rounds="$(config_rounds "${config}")"
  clients="$(config_clients "${config}")"
  total_updates=$((rounds * clients))
  local last_count=-1
  local last_eval=-1

  while kill -0 "${pid}" 2>/dev/null; do
    local count=0
    if [[ -d "${root}" ]]; then
      count="$(find "${root}" -type f -path '*/updates/*.npz' 2>/dev/null | wc -l | tr -d ' ')"
    fi

    if [[ "${count}" =~ ^[0-9]+$ ]] && (( count > 0 && count != last_count )); then
      local round client pct
      round=$(( (count - 1) / clients + 1 ))
      client=$(( (count - 1) % clients + 1 ))
      pct=$(( count * 100 / total_updates ))
      echo "[$(date '+%F %T')] PROGRESS GPU=${gpu} ${job_label} round=${round}/${rounds} client=${client}/${clients} total=${count}/${total_updates} (${pct}%)"
      last_count="${count}"
    fi

    local metrics
    metrics="$(find "${root}" -name round_metrics.csv -print -quit 2>/dev/null || true)"
    if [[ -n "${metrics}" && -s "${metrics}" ]]; then
      local eval_round
      eval_round="$(awk -F, 'NR>1 {r=$2} END {if (r!="") print r}' "${metrics}")"
      if [[ "${eval_round}" =~ ^[0-9]+$ ]] && (( eval_round != last_eval )); then
        awk -F, -v gpu="${gpu}" -v label="${job_label}" '
          NR>1 {r=$2; va=$4; vf=$5; vu=$9; ta=$11; tf=$12; tu=$16}
          END {
            if (r != "") {
              cmd="date +\"%F %T\""; cmd | getline now; close(cmd);
              printf("[%s] METRICS  GPU=%s %s round=%s val_acc=%s val_f1=%s val_auroc=%s test_acc=%s test_f1=%s test_auroc=%s\n", now,gpu,label,r,va,vf,vu,ta,tf,tu);
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

make_smoke_config() {
  local source_config="$1"
  local smoke_dir="${GENERATED_ROOT}/smoke"
  local smoke_config="${smoke_dir}/E1_hateful_smoke.yaml"
  mkdir -p "${smoke_dir}"
  python - "${source_config}" "${smoke_config}" <<'PY'
import sys, yaml
src, dst = sys.argv[1], sys.argv[2]
with open(src, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
cfg["federated"]["num_clients"] = 2
cfg["federated"]["samples_per_client"] = 20
cfg["federated"]["rounds"] = 2
cfg["federated"]["min_participants"] = 2
cfg["evaluation"]["eval_every"] = 1
cfg["experiment"]["job_id"] = "E1_hateful_smoke"
cfg["experiment"]["output_root"] = "results/acm_revision/smoke/E1_hateful_smoke"
cfg["update_capture"]["save_all_checkpoints"] = False
cfg["update_capture"]["checkpoint_rounds"] = []
cfg["update_capture"]["groups"] = []
with open(dst, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY
  echo "${smoke_config}"
}

run_one() {
  local config="$1"
  local gpu="$2"
  local job_pos="${3:-?}"
  local job_total="${4:-?}"
  local job_id log root meta
  job_id="$(basename "${config}" .yaml)"
  log="${LOG_ROOT}/${job_id}.log"
  root="$(config_output_root "${config}")"
  meta="$(config_meta "${config}")"

  if job_done "${config}"; then
    echo "[$(date '+%F %T')] SKIP  GPU=${gpu} job=${job_pos}/${job_total} ${job_id} ${meta} already complete"
    cleanup_heavy_artifacts "${config}"
    return 0
  fi

  # A previous interrupted attempt must not contaminate progress counters/results.
  if [[ -d "${root}" ]]; then
    rm -rf "${root}"
  fi

  echo "[$(date '+%F %T')] START GPU=${gpu} job=${job_pos}/${job_total} ${job_id} ${meta}"
  echo "[$(date '+%F %T')] LOG   ${log}"

  PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${gpu}" \
    python -m src.acm_revision.cli run-fl --config "${config}" > "${log}" 2>&1 &
  local train_pid=$!

  progress_monitor "${train_pid}" "${config}" "${gpu}" "job=${job_pos}/${job_total} ${meta}" &
  local monitor_pid=$!

  local status=0
  if ! wait "${train_pid}"; then
    status=$?
  fi
  wait "${monitor_pid}" 2>/dev/null || true

  if (( status == 0 )) && job_done "${config}"; then
    cleanup_heavy_artifacts "${config}"
    echo "[$(date '+%F %T')] DONE  GPU=${gpu} job=${job_pos}/${job_total} ${job_id} ${meta}"
    return 0
  fi

  echo "[$(date '+%F %T')] FAIL  GPU=${gpu} job=${job_pos}/${job_total} ${job_id} ${meta}" >&2
  echo "---------------- tail ${log} ----------------" >&2
  tail -n 80 "${log}" >&2 || true
  echo "------------------------------------------------" >&2
  return 1
}

if [[ "${MODE}" == "smoke" ]]; then
  smoke_config="$(make_smoke_config "${CONFIGS[0]}")"
  echo "[INFO] True smoke test: 2 clients x 20 samples x 2 rounds on GPU ${GPUS[0]}"
  echo "[INFO] Smoke output is isolated from the 200 formal E1 jobs."
  run_one "${smoke_config}" "${GPUS[0]}" 1 1
  exit $?
fi

echo "[INFO] Full E1 Hateful: 200 formal jobs, each 25 rounds."
if [[ ${#GPUS[@]} -eq 2 ]]; then
  echo "[INFO] Seed split: GPU ${GPUS[0]} <- seeds 42-46; GPU ${GPUS[1]} <- seeds 47-51"
  echo "[INFO] Each GPU runs 100 jobs sequentially. Live client/round progress will print here."
fi

run_seed_worker() {
  local gpu="$1"
  local seed_min="$2"
  local seed_max="$3"
  local failures=0
  local config seed
  local pos=0
  local total=100

  echo "[$(date '+%F %T')] WORKER GPU=${gpu} seeds=${seed_min}-${seed_max} total_jobs=${total}"
  for config in "${CONFIGS[@]}"; do
    seed="$(config_seed "${config}")"
    if [[ -z "${seed}" ]]; then
      echo "[ERROR] Could not read seed from ${config}" >&2
      failures=$((failures + 1))
      continue
    fi
    if (( seed < seed_min || seed > seed_max )); then
      continue
    fi
    pos=$((pos + 1))
    if ! run_one "${config}" "${gpu}" "${pos}" "${total}"; then
      failures=$((failures + 1))
    fi
  done

  if (( failures > 0 )); then
    echo "[ERROR] GPU ${gpu} worker had ${failures} failed job(s)." >&2
    return 1
  fi
  echo "[$(date '+%F %T')] WORKER DONE GPU=${gpu} seeds=${seed_min}-${seed_max}"
}

# Preferred two-GPU protocol: split independent FL seeds 5 + 5.
if [[ ${#GPUS[@]} -eq 2 ]]; then
  failures=0
  run_seed_worker "${GPUS[0]}" 42 46 &
  pid0=$!
  run_seed_worker "${GPUS[1]}" 47 51 &
  pid1=$!

  if ! wait "${pid0}"; then
    failures=$((failures + 1))
  fi
  if ! wait "${pid1}"; then
    failures=$((failures + 1))
  fi

  if (( failures > 0 )); then
    echo "[ERROR] ${failures} GPU worker(s) reported failures. Re-run the same command; completed jobs will be skipped." >&2
    exit 1
  fi

  echo "[OK] Full E1 Hateful run finished successfully."
  exit 0
fi

# Generic fallback for one GPU or more than two GPUs: one job at a time per GPU.
active=0
failures=0
for idx in "${!CONFIGS[@]}"; do
  config="${CONFIGS[$idx]}"
  gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
  run_one "${config}" "${gpu}" "$((idx + 1))" "${#CONFIGS[@]}" &
  active=$((active + 1))
  if (( active >= ${#GPUS[@]} )); then
    if ! wait -n; then
      failures=$((failures + 1))
    fi
    active=$((active - 1))
  fi
done

while (( active > 0 )); do
  if ! wait -n; then
    failures=$((failures + 1))
  fi
  active=$((active - 1))
done

if (( failures > 0 )); then
  echo "[ERROR] ${failures} job(s) failed. Re-run the same command after fixing the first error; completed jobs will be skipped." >&2
  exit 1
fi

echo "[OK] Full E1 Hateful run finished successfully."
