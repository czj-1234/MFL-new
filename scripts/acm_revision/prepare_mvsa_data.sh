#!/usr/bin/env bash
set -euo pipefail

# Prepare strict Shadow-Train / Shadow-Val / Target pools for MVSA-Multiple.
# Exact duplicate identities/pairs are removed. Repeated/near-duplicate content
# is retained in indivisible families. Splitting is stratified jointly by
# (image_label, text_label) so image-only/text-only/modality-exclusive regimes
# share comparable population distributions.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="configs/acm_revision/mvsa_multiple.yaml"
REPORT="data/processed/acm_revision/mvsa_multiple/split_report.json"

if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    echo "[ERROR] Neither python nor python3 was found in PATH." >&2
    exit 1
fi

for required in \
    data/processed/mvsa_3class_train.json \
    data/processed/mvsa_3class_val.json \
    data/processed/mvsa_3class_test.json; do
    if [[ ! -f "${required}" ]]; then
        echo "[ERROR] Missing ${required}" >&2
        exit 1
    fi
done

echo "============================================================"
echo "ACM Revision - MVSA-Multiple strict data preparation"
echo "Policy     : exact dedup + grouped near-duplicates + joint-label stratification"
echo "Repository : ${REPO_ROOT}"
echo "Config     : ${CONFIG}"
echo "Python     : $(${PYTHON_BIN} --version 2>&1)"
echo "============================================================"

${PYTHON_BIN} -m src.acm_revision.prepare_data --config "${CONFIG}"

echo
if [[ -f "${REPORT}" ]]; then
    echo "================ split_report.json ========================="
    cat "${REPORT}"
    echo
    echo "============================================================"
    echo "[OK] Strict MVSA-Multiple pools were generated successfully."
    echo "Output: data/processed/acm_revision/mvsa_multiple/"
else
    echo "[ERROR] Command finished but ${REPORT} was not created." >&2
    exit 1
fi
