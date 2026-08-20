#!/usr/bin/env bash
set -euo pipefail

export DEFENSE70_MIN_FREE_MIB="${DEFENSE70_MIN_FREE_MIB:-23000}"
export DEFENSE70_STARTUP_WAIT="${DEFENSE70_STARTUP_WAIT:-90}"
export DEFENSE70_MAX_RETRIES="${DEFENSE70_MAX_RETRIES:-2}"
export DEFENSE70_RETRY_SLEEP="${DEFENSE70_RETRY_SLEEP:-60}"
export MFL_NFS_WRITE_RETRIES="${MFL_NFS_WRITE_RETRIES:-8}"

exec bash scripts/acm_revision/run_defense70_server.sh A "${1:-0}" "${2:-1}"
