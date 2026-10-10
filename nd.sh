#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python scripts/acm_revision/audit_headfix.py \
  --root results/acm_revision/core72_modality_first_v2_r150 \
  --out results/acm_revision/nodefense_worstcase_audit
