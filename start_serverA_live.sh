#!/usr/bin/env bash
# Server A: start its eight ASLWF FL runs on two GPUs and print live progress.
# Use: bash start_serverA_live.sh   (GPU0=0 GPU1=1 by default)
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
exec bash scripts/acm_revision/serverA_uncentered_8.sh "$@"
