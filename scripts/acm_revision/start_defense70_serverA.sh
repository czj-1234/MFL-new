#!/usr/bin/env bash
set -euo pipefail

GPU0="${1:-0}"
GPU1="${2:-1}"
RUNTIME="logs/acm_revision/defense70_r150/runtime"
mkdir -p "${RUNTIME}"

PIDFILE="${RUNTIME}/serverA.pid"
MASTERLOG="${RUNTIME}/serverA.nohup.log"

if [[ -f "${PIDFILE}" ]]; then
  OLD_PID="$(cat "${PIDFILE}" 2>/dev/null || true)"
  if [[ "${OLD_PID}" =~ ^[0-9]+$ ]] && kill -0 "${OLD_PID}" 2>/dev/null; then
    echo "Server A already running: PID=${OLD_PID}"
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
echo "Check: tail -f ${MASTERLOG}"
