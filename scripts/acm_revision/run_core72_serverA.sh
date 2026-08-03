#!/usr/bin/env bash
set -euo pipefail
echo "[INFO] Redirecting to isolated modality-first Core72 V2 runner." >&2
exec bash scripts/acm_revision/run_core72_modality_first_serverA.sh "${1:-0}" "${2:-1}"
