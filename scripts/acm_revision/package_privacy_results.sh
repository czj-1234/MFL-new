#!/usr/bin/env bash
set -euo pipefail

ROUNDS="${1:-150}"
HOST_TAG="${2:-$(hostname -s)}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="privacy_results_${HOST_TAG}_r${ROUNDS}_${STAMP}.tar.gz"

paths=()
for path in \
  "results/acm_revision/privacy_capture_r${ROUNDS}" \
  "results/acm_revision/privacy_postprocess_r${ROUNDS}" \
  "logs/acm_revision/privacy_capture_r${ROUNDS}" \
  "configs/acm_revision/generated/privacy_capture_r${ROUNDS}"; do
  [[ -e "${path}" ]] && paths+=("${path}")
done

if (( ${#paths[@]} == 0 )); then
  echo "No privacy results found for r${ROUNDS}." >&2
  exit 1
fi

tar -czf "${OUT}" "${paths[@]}"
echo "Created ${OUT}"
