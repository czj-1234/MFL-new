#!/usr/bin/env bash
set -euo pipefail

ROOT="${DEFENSE70_LOG_ROOT:-logs/acm_revision/defense70_r150}"
STATE="${DEFENSE70_STATE_ROOT:-${ROOT}/state}"
RUNTIME="${ROOT}/runtime"
CHECKPOINT_ROOT="${DEFENSE70_CHECKPOINT_ROOT:-checkpoints/acm_revision/defense70_r150}"

DONE=0
FAILED=0
[[ -d "${STATE}/done" ]] && DONE="$(find "${STATE}/done" -maxdepth 1 -type f -name '*.done' | wc -l)"
[[ -d "${STATE}/failed" ]] && FAILED="$(find "${STATE}/failed" -maxdepth 1 -type f -name '*.failed' | wc -l)"

echo "Defense70 status"
echo "---------------"
echo "PASS jobs      : ${DONE}/70"
echo "FAILED markers : ${FAILED}"
echo "Remaining      : $((70 - DONE))"
echo "Checkpoint root: ${CHECKPOINT_ROOT}"
echo "Checkpoint step: ${DEFENSE70_CHECKPOINT_EVERY:-15} rounds"

for S in A B; do
  PIDFILE="${RUNTIME}/server${S}.pid"
  if [[ -f "${PIDFILE}" ]]; then
    PID="$(cat "${PIDFILE}" 2>/dev/null || true)"
    if [[ "${PID}" =~ ^[0-9]+$ ]] && kill -0 "${PID}" 2>/dev/null; then
      echo "Server ${S}: RUNNING PID=${PID}"
    else
      echo "Server ${S}: NOT RUNNING (stale pidfile)"
    fi
  else
    echo "Server ${S}: no pidfile"
  fi
done

echo
echo "Latest external resume checkpoints:"
find "${CHECKPOINT_ROOT}" -type f -name 'round_*.pt' 2>/dev/null | sort | tail -n 30 || true

echo
echo "GPU processes:"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader 2>/dev/null || true
