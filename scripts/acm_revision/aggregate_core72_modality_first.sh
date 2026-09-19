#!/usr/bin/env bash
set -euo pipefail

if (( $# < 1 || $# > 2 )); then
  echo "Usage: $0 <postprocess-root-A-or-merged-root> [postprocess-root-B]" >&2
  exit 2
fi

export CORE72_FINAL_ROOT="${CORE72_FINAL_ROOT:-results/acm_revision/core72_modality_first_v2_final}"
exec bash scripts/acm_revision/aggregate_core72.sh "$@"
