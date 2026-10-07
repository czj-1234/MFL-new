#!/usr/bin/env bash
set -euo pipefail

REFERENCE_CONFIG="${1:-configs/acm_revision/generated/defense70_prep/reference_seed142.yaml}"
REFERENCE_RUN_DIR="${2:-}"
HEADFIX_OUT="${3:-results/acm_revision/headfix_prep}"

echo "[HEADFIX PREP] reference_config=$REFERENCE_CONFIG"
echo "[HEADFIX PREP] output_dir=$HEADFIX_OUT"

if [[ -n "$REFERENCE_RUN_DIR" ]]; then
  python -m src.acm_revision.headfix_prepare \
    --reference-config "$REFERENCE_CONFIG" \
    --reference-run-dir "$REFERENCE_RUN_DIR" \
    --output-dir "$HEADFIX_OUT"
else
  python -m src.acm_revision.headfix_prepare \
    --reference-config "$REFERENCE_CONFIG" \
    --output-dir "$HEADFIX_OUT"
fi

python -m src.acm_revision.headfix_plan \
  --head-manifest "$HEADFIX_OUT/locked_headfix.yaml" \
  --old-manifest results/acm_revision/defense70_prep/locked_params.yaml \
  --output-dir configs/acm_revision/generated/headfix \
  --base-config configs/acm_revision/hateful_memes.yaml

echo "[HEADFIX PREP] generated queues:"
ls -1 configs/acm_revision/generated/headfix/server*_gpu*.tsv
