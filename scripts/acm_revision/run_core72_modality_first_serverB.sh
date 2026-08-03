#!/usr/bin/env bash
set -euo pipefail
exec bash scripts/acm_revision/run_core72_modality_first_server.sh B "${1:-0}" "${2:-1}"
