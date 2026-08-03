#!/usr/bin/env bash
set -euo pipefail
echo "[INFO] Redirecting to isolated modality-first Core72 V2 smoke test." >&2
exec bash scripts/acm_revision/smoke_test_core72_modality_first.sh "${1:-0}"
