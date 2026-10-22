#!/usr/bin/env bash
# Server B: start its eight ASLWF FL runs on two GPUs and print live progress.
# Use: bash start_serverB_live.sh   (GPU0=0 GPU1=1 by default)
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
exec bash scripts/acm_revision/serverB_c09_8.sh "$@"
