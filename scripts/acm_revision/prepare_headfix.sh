#!/usr/bin/env bash
set -euo pipefail

CORE_ROOT="${1:-results/acm_revision/core72_r150}"
HEADFIX_OUT="${2:-results/acm_revision/headfix_prep}"
MAX_RANK="${3:-64}"
TARGET_AUC="${4:-0.70}"
LAUNCH_AUC="${5:-0.80}"

echo "[HEADFIX PREP] existing Core72 root=$CORE_ROOT"
echo "[HEADFIX PREP] output_dir=$HEADFIX_OUT"
echo "[HEADFIX PREP] max_rank=$MAX_RANK target_auc=$TARGET_AUC launch_auc=$LAUNCH_AUC"
echo "[HEADFIX PREP] no CLIP/FL retraining is performed in this stage"

python -m src.acm_revision.headfix_prepare \
  --core-root "$CORE_ROOT" \
  --output-dir "$HEADFIX_OUT" \
  --max-rank "$MAX_RANK" \
  --target-auc "$TARGET_AUC" \
  --launch-auc "$LAUNCH_AUC"

STATUS="$(python - <<PY
import yaml
with open("$HEADFIX_OUT/locked_headfix.yaml", "r", encoding="utf-8") as f:
    print(yaml.safe_load(f)["status"])
PY
)"

if [[ "$STATUS" != "LOCKED" ]]; then
  echo "[HEADFIX PREP] status=$STATUS; formal 8-run matrix was NOT generated."
  echo "[HEADFIX PREP] inspect $HEADFIX_OUT/headfix_candidates.json"
  exit 3
fi

python -m src.acm_revision.headfix_plan \
  --head-manifest "$HEADFIX_OUT/locked_headfix.yaml" \
  --old-manifest results/acm_revision/defense70_prep/locked_params.yaml \
  --output-dir configs/acm_revision/generated/headfix \
  --base-config configs/acm_revision/hateful_memes.yaml

echo "[HEADFIX PREP] status=LOCKED; generated queues:"
ls -1 configs/acm_revision/generated/headfix/server*_gpu*.tsv
