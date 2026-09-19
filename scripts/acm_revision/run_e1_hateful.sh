#!/usr/bin/env bash
set -uo pipefail

GPU_LIST="${1:-0}"
MODE="${2:-all}"   # all | smoke
MATRIX="configs/acm_revision/e1_hateful.yaml"
PROFILE="E1_task_performance_hateful"
GENERATED_ROOT="configs/acm_revision/generated"
LOG_ROOT="logs/acm_revision/${PROFILE}"
RESULT_ROOT="results/acm_revision/${PROFILE}"
GPU_MIN_FREE_MIB="${GPU_MIN_FREE_MIB:-16000}"

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
  awk '/^seed:[[:space:]]*/ {print $2; exit}' "$1"
}

config_output_root() {
  python - "$1" <<'PY'
import sys, yaml
with open(sys.argv[1], "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
print(cfg["experiment"]["output_root"])
PY
}

config_meta() {
  python - "$1" <<'PY'
import sys, yaml
with open(sys.argv[1], "r", encoding="utf-8") as f:
    c = yaml.safe_load(f)
print(
    f"seed={c['seed']} "
    f"setting={c['experiment']['setting_name']} "
    f"c={c['experiment']['concentration']} "
    f"rounds={c['federated']['rounds']} "
    f"clients={c['federated']['num_clients']} "
    f"batch={c['federated']['batch_size']} "
    f"local_epochs={c['federated']['local_epochs']}"
)
PY
}

config_rounds() {
  python - "$1" <<'PY'
import sys, yaml
with open(sys.argv[1], "r", encoding="utf-8") as f:
    print(int(yaml.safe_load(f)["federated"]["rounds"]))
PY
}

config_clients() {
  python - "$1" <<'PY'
import sys, yaml
with open(sys.argv[1], "r", encoding="utf-8") as f:
    print(int(yaml.safe_load(f)["federated"]["num_clients"]))
PY
}

job_done() {
  local root
  root="$(config_output_root "$1")"
  [[ -d "${root}" ]] && find "${root}" -name summary.json -print -quit 2>/dev/null | grep -q .
}

cleanup_heavy_artifacts() {
  local root
  root="$(config_output_root "$1")"
  [[ -d "${root}" ]] || return 0
  find "${root}" -type f -name 'best_model.pt' -delete 2>/dev/null || true
  find "${root}" -type d -name checkpoints -prune -exec rm -rf {} + 2>/dev/null || true
  # E1 has no attack-vector groups. Update NPZs are only live progress markers.
  find "${root}" -type d -name updates -prune -exec rm -rf {} + 2>/dev/null || true
}

wait_for_gpu_free() {
  local gpu="$1"
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  while true; do
    local free_mib
    free_mib="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -dc '0-9')"
    if [[ "${free_mib}" =~ ^[0-9]+$ ]] && (( free_mib >= GPU_MIN_FREE_MIB )); then
      echo "[$(date '+%F %T')] GPU=${gpu} ready: free=${free_mib} MiB (required >= ${GPU_MIN_FREE_MIB} MiB)"
      return 0
    fi
    echo "[$(date '+%F %T')] WAIT GPU=${gpu}: free=${free_mib:-unknown} MiB; waiting for >= ${GPU_MIN_FREE_MIB} MiB to avoid OOM"
    sleep 30
  done
}

progress_monitor() {
  local pid="$1" config="$2" gpu="$3" job_label="$4"
  local root rounds clients total_updates last_count=-1 last_eval=-1
  root="$(config_output_root "${config}")"
  rounds="$(config_rounds "${config}")"
  clients="$(config_clients "${config}")"
  total_updates=$((rounds * clients))

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
        tail -n 1 "${metrics}" | awk -F, -v gpu="${gpu}" -v label="${job_label}" '{
          cmd="date +\"%F %T\""; cmd | getline now; close(cmd);
          printf("[%s] METRICS GPU=%s %s round=%s val_acc=%s val_f1=%s val_auroc=%s test_acc=%s test_f1=%s test_auroc=%s\n", now,gpu,label,$2,$4,$5,$9,$11,$12,$16);
          fflush();
        }'
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

CURRENT_TRAIN_PID=""
CURRENT_MONITOR_PID=""
stop_current_job() {
  if [[ -n "${CURRENT_MONITOR_PID:-}" ]]; then
    kill "${CURRENT_MONITOR_PID}" 2>/dev/null || true
  fi
  if [[ -n "${CURRENT_TRAIN_PID:-}" ]]; then
    pkill -TERM -P "${CURRENT_TRAIN_PID}" 2>/dev/null || true
    kill -TERM "${CURRENT_TRAIN_PID}" 2>/dev/null || true
    sleep 1
    pkill -KILL -P "${CURRENT_TRAIN_PID}" 2>/dev/null || true
    kill -KILL "${CURRENT_TRAIN_PID}" 2>/dev/null || true
  fi
  CURRENT_TRAIN_PID=""
  CURRENT_MONITOR_PID=""
}

run_one() {
  local config="$1" gpu="$2" label="$3"
  local job_id log root meta
  job_id="$(basename "${config}" .yaml)"
  log="${LOG_ROOT}/${job_id}.log"
  root="$(config_output_root "${config}")"
  meta="$(config_meta "${config}")"

  if job_done "${config}"; then
    echo "[$(date '+%F %T')] SKIP GPU=${gpu} ${label} ${job_id} ${meta} already complete"
    cleanup_heavy_artifacts "${config}"
    return 0
  fi

  [[ -d "${root}" ]] && rm -rf "${root}"
  wait_for_gpu_free "${gpu}"

  echo "[$(date '+%F %T')] START GPU=${gpu} ${label} ${job_id} ${meta}"
  echo "[$(date '+%F %T')] LOG   ${log}"

  PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${gpu}" \
    python -m src.acm_revision.cli run-fl --config "${config}" > "${log}" 2>&1 &
  CURRENT_TRAIN_PID=$!
  progress_monitor "${CURRENT_TRAIN_PID}" "${config}" "${gpu}" "${label} ${meta}" &
  CURRENT_MONITOR_PID=$!

  local status=0
  wait "${CURRENT_TRAIN_PID}" || status=$?
  wait "${CURRENT_MONITOR_PID}" 2>/dev/null || true
  CURRENT_TRAIN_PID=""
  CURRENT_MONITOR_PID=""

  if (( status == 0 )) && job_done "${config}"; then
    cleanup_heavy_artifacts "${config}"
    echo "[$(date '+%F %T')] DONE GPU=${gpu} ${label} ${job_id} ${meta}"
    return 0
  fi

  echo "[$(date '+%F %T')] FAIL GPU=${gpu} ${label} ${job_id} ${meta}" >&2
  echo "---------------- tail ${log} ----------------" >&2
  tail -n 80 "${log}" >&2 || true
  echo "------------------------------------------------" >&2
  return 1
}

if [[ "${MODE}" == "smoke" ]]; then
  trap 'stop_current_job' INT TERM EXIT
  smoke_config="$(make_smoke_config "${CONFIGS[0]}")"
  echo "[INFO] True smoke test: 2 clients x 20 samples x 2 rounds on GPU ${GPUS[0]}"
  run_one "${smoke_config}" "${GPUS[0]}" "smoke=1/1"
  status=$?
  trap - INT TERM EXIT
  exit ${status}
fi

echo "[INFO] Full E1 Hateful: 10 seeds x 20 jobs/seed = 200 jobs; each formal job is 25 rounds."
echo "[INFO] Seed barrier enabled: a GPU cannot advance to its next seed until all 20 jobs of the current seed are complete."
echo "[INFO] Fail-fast enabled: if one job fails, that GPU worker stops on the current seed. Re-run the same command after fixing the cause."
echo "[INFO] GPU preflight enabled: a new job waits until at least ${GPU_MIN_FREE_MIB} MiB is free."

run_seed_worker() {
  local gpu="$1" seed_min="$2" seed_max="$3"
  trap 'stop_current_job; exit 130' INT TERM

  local seed config config_seed_value local_pos seed_complete
  for ((seed=seed_min; seed<=seed_max; seed++)); do
    mapfile -t SEED_CONFIGS < <(
      for config in "${CONFIGS[@]}"; do
        config_seed_value="$(config_seed "${config}")"
        [[ "${config_seed_value}" == "${seed}" ]] && echo "${config}"
      done
    )

    if [[ ${#SEED_CONFIGS[@]} -ne 20 ]]; then
      echo "[ERROR] GPU=${gpu} seed=${seed}: expected 20 jobs, found ${#SEED_CONFIGS[@]}." >&2
      return 1
    fi

    echo "============================================================"
    echo "[$(date '+%F %T')] SEED START GPU=${gpu} seed=${seed} jobs=20"
    echo "============================================================"

    local_pos=0
    for config in "${SEED_CONFIGS[@]}"; do
      local_pos=$((local_pos + 1))
      if ! run_one "${config}" "${gpu}" "seed=${seed} seed_job=${local_pos}/20"; then
        echo "[ERROR] GPU=${gpu} seed=${seed} BLOCKED at seed_job=${local_pos}/20." >&2
        echo "[ERROR] This worker will NOT advance to seed=$((seed + 1)). Fix the failure and re-run; completed jobs will be skipped." >&2
        return 1
      fi
    done

    seed_complete=0
    for config in "${SEED_CONFIGS[@]}"; do
      if job_done "${config}"; then
        seed_complete=$((seed_complete + 1))
      fi
    done
    if (( seed_complete != 20 )); then
      echo "[ERROR] GPU=${gpu} seed=${seed}: only ${seed_complete}/20 jobs have summary.json; refusing to advance." >&2
      return 1
    fi

    echo "============================================================"
    echo "[$(date '+%F %T')] SEED COMPLETE GPU=${gpu} seed=${seed} 20/20"
    echo "============================================================"
  done

  echo "[$(date '+%F %T')] WORKER DONE GPU=${gpu} seeds=${seed_min}-${seed_max}"
  return 0
}

if [[ ${#GPUS[@]} -eq 2 ]]; then
  echo "[INFO] GPU ${GPUS[0]}: seed42 -> 43 -> 44 -> 45 -> 46"
  echo "[INFO] GPU ${GPUS[1]}: seed47 -> 48 -> 49 -> 50 -> 51"

  run_seed_worker "${GPUS[0]}" 42 46 &
  pid0=$!
  run_seed_worker "${GPUS[1]}" 47 51 &
  pid1=$!

  cleanup_parent() {
    kill -TERM "${pid0}" "${pid1}" 2>/dev/null || true
    wait "${pid0}" "${pid1}" 2>/dev/null || true
  }
  trap 'cleanup_parent' INT TERM

  failures=0
  wait "${pid0}" || failures=$((failures + 1))
  wait "${pid1}" || failures=$((failures + 1))
  trap - INT TERM

  if (( failures > 0 )); then
    echo "[ERROR] ${failures} GPU worker(s) stopped. Completed jobs remain valid; re-run the same command after fixing the cause." >&2
    exit 1
  fi
  echo "[OK] Full E1 Hateful run finished successfully."
  exit 0
fi

if [[ ${#GPUS[@]} -eq 1 ]]; then
  run_seed_worker "${GPUS[0]}" 42 51
  exit $?
fi

echo "[ERROR] This formal seed-barrier runner currently supports one GPU or exactly two GPUs." >&2
exit 2
