#!/usr/bin/env bash
set -u

PROFILE="E1_task_performance_hateful"
RESULT_ROOT="results/acm_revision/${PROFILE}"
LOG_ROOT="logs/acm_revision/${PROFILE}"

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

echo "============================================================"
echo " E1 Hateful status check"
echo "============================================================"
echo "Project: $(pwd)"
echo "Time:    $(date '+%F %T')"
echo

if [[ ! -d "${RESULT_ROOT}" ]]; then
  echo "[ERROR] Result directory not found: ${RESULT_ROOT}"
  exit 1
fi

mapfile -t JOB_DIRS < <(find "${RESULT_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'E1_task_performance_hateful_*' | sort)

complete=0
partial=0
empty=0
first_partial=""
last_partial=""

echo "[1] Job directory status"
for d in "${JOB_DIRS[@]}"; do
  name="$(basename "$d")"
  if find "$d" -name summary.json -print -quit 2>/dev/null | grep -q .; then
    complete=$((complete + 1))
    printf "OK       %s\n" "$name"
  elif [[ -f "$d/resolved_config.json" || -f "$d/client_partition_manifest.json" ]] || \
       find "$d" -name resolved_config.json -print -quit 2>/dev/null | grep -q .; then
    partial=$((partial + 1))
    [[ -z "$first_partial" ]] && first_partial="$name"
    last_partial="$name"
    printf "PARTIAL  %s\n" "$name"
  else
    empty=$((empty + 1))
    printf "EMPTY    %s\n" "$name"
  fi
done

echo
echo "Summary: directories=${#JOB_DIRS[@]} complete=${complete} partial=${partial} empty=${empty}"

if (( complete > 0 )); then
  echo
  echo "[2] Completed jobs: ${complete}"
fi

if (( partial > 0 )); then
  echo "[3] Partial jobs:   ${partial}"
  echo "    First partial: ${first_partial}"
  echo "    Last partial:  ${last_partial}"
fi

echo
echo "[4] Active FL processes"
ACTIVE="$(ps -ef | grep 'src.acm_revision.cli run-fl' | grep -v grep || true)"
if [[ -n "$ACTIVE" ]]; then
  echo "$ACTIVE"
else
  echo "No active run-fl process found."
fi

echo
echo "[5] GPU status"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || nvidia-smi
else
  echo "nvidia-smi not found."
fi

show_log_tail() {
  local job="$1"
  local log="${LOG_ROOT}/${job}.log"
  echo
  echo "------------------------------------------------------------"
  echo "Log for ${job}"
  echo "------------------------------------------------------------"
  if [[ -f "$log" ]]; then
    echo "Path: $log"
    echo
    tail -n 100 "$log"
  else
    echo "[WARN] Log file not found: $log"
  fi
}

if [[ -n "$first_partial" ]]; then
  echo
echo "[6] First partial job log tail"
  show_log_tail "$first_partial"
fi

if [[ -n "$last_partial" && "$last_partial" != "$first_partial" ]]; then
  echo
echo "[7] Last partial job log tail"
  show_log_tail "$last_partial"
fi

echo
echo "[8] Error signatures from partial-job logs"
found_error=0
if [[ -d "${LOG_ROOT}" ]]; then
  for d in "${JOB_DIRS[@]}"; do
    job="$(basename "$d")"
    if find "$d" -name summary.json -print -quit 2>/dev/null | grep -q .; then
      continue
    fi
    log="${LOG_ROOT}/${job}.log"
    [[ -f "$log" ]] || continue
    err="$(grep -Eai 'Traceback|CUDA out of memory|OutOfMemoryError|RuntimeError|ValueError|KeyError|FileNotFoundError|Exception|Killed|No space left|failed|error' "$log" | tail -n 3 || true)"
    if [[ -n "$err" ]]; then
      found_error=1
      echo
      echo "### ${job}"
      echo "$err"
    fi
  done
fi
if (( found_error == 0 )); then
  echo "No obvious error signature found in partial-job logs."
fi

echo
echo "============================================================"
echo " End of check"
echo "============================================================"
