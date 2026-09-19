#!/usr/bin/env bash
set -euo pipefail

GPU0="${1:-0}"
GPU1="${2:-1}"
RUNTIME="logs/acm_revision/defense70_r150/runtime"
mkdir -p "${RUNTIME}"

PIDFILE="${RUNTIME}/serverA.pid"
MASTERLOG="${RUNTIME}/serverA.nohup.log"
MONITOR="scripts/acm_revision/watch_defense70_progress.sh"

attach_monitor() {
  if [[ -t 1 && -f "${MONITOR}" ]]; then
    echo "Opening live Defense70 progress view..."
    echo "Ctrl+C closes only the progress view; background training keeps running."
    exec bash "${MONITOR}" A
  fi
}

if [[ -f "${PIDFILE}" ]]; then
  OLD_PID="$(cat "${PIDFILE}" 2>/dev/null || true)"
  if [[ "${OLD_PID}" =~ ^[0-9]+$ ]] && kill -0 "${OLD_PID}" 2>/dev/null; then
    echo "Server A already running: PID=${OLD_PID}"
    attach_monitor
    exit 0
  fi
fi

nohup setsid bash scripts/acm_revision/run_defense70_serverA.sh "${GPU0}" "${GPU1}" \
  > "${MASTERLOG}" 2>&1 < /dev/null &
PID=$!
echo "${PID}" > "${PIDFILE}"

echo "Server A started safely in background."
echo "PID=${PID}"
echo "Master log: ${MASTERLOG}"
echo "Background training is protected from SSH disconnect."

# Give the scheduler a moment to validate jobs.tsv and launch workers.
sleep 2
attach_monitor

echo "Live monitor not attached (non-interactive shell)."
echo "Use: bash scripts/acm_revision/watch_defense70_progress.sh A"
