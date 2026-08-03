#!/usr/bin/env bash
set -euo pipefail
echo "[INFO] Redirecting legacy Core72 runner to isolated modality-first V2 namespace." >&2
exec bash scripts/acm_revision/run_core72_modality_first_server.sh "$@"
