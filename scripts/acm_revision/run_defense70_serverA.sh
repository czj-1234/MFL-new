#!/usr/bin/env bash
set -euo pipefail

export DEFENSE70_MIN_FREE_MIB="${DEFENSE70_MIN_FREE_MIB:-23000}"
export DEFENSE70_STARTUP_WAIT="${DEFENSE70_STARTUP_WAIT:-90}"
export DEFENSE70_MAX_RETRIES="${DEFENSE70_MAX_RETRIES:-2}"
export DEFENSE70_RETRY_SLEEP="${DEFENSE70_RETRY_SLEEP:-60}"
export DEFENSE70_CHECKPOINT_ROOT="${DEFENSE70_CHECKPOINT_ROOT:-checkpoints/acm_revision/defense70_r150}"
export DEFENSE70_CHECKPOINT_EVERY="${DEFENSE70_CHECKPOINT_EVERY:-15}"
export DEFENSE70_CHECKPOINT_KEEP_LAST="${DEFENSE70_CHECKPOINT_KEEP_LAST:-2}"
export MFL_NFS_WRITE_RETRIES="${MFL_NFS_WRITE_RETRIES:-8}"

GPU0="${1:-0}"
GPU1="${2:-1}"

# Server A owns the one-time scientific preparation. This step is idempotent:
# if the locked shadow-only manifest and 70 generated configs already pass the
# audit, it returns immediately. Otherwise it runs/resumes the shadow reference,
# fits the bases, tunes on shadow_val only, freezes parameters, then generates
# jobs.tsv. No target data are used here.
bash scripts/acm_revision/prepare_defense70.sh "${GPU0}"

exec bash scripts/acm_revision/run_defense70_server.sh A "${GPU0}" "${GPU1}"
