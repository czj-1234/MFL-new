#!/usr/bin/env bash
set -euo pipefail

# Prepare strict Shadow-Train / Shadow-Val / Target pools for Hateful Memes.
# This version preserves the original training examples and prevents leakage by
# assigning exact/near-duplicate families to a single strict pool.
# Usage from anywhere inside the repository:
#   bash scripts/acm_revision/prepare_hateful_data.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="configs/acm_revision/hateful_memes.yaml"
REPORT="data/processed/acm_revision/hateful_memes/split_report.json"

if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    echo "[ERROR] Neither python nor python3 was found in PATH." >&2
    exit 1
fi

echo "============================================================"
echo "ACM Revision - Hateful Memes strict data preparation"
echo "Policy     : preserve all raw examples; group duplicate families"
echo "Repository : ${REPO_ROOT}"
echo "Config     : ${CONFIG}"
echo "Python     : $(${PYTHON_BIN} --version 2>&1)"
echo "============================================================"

if [[ ! -f "${CONFIG}" ]]; then
    echo "[ERROR] Missing config: ${CONFIG}" >&2
    echo "Make sure you are on branch: acm-transactions-revision" >&2
    exit 1
fi

for required in \
    data/processed/hateful_train.json \
    data/processed/hateful_val.json \
    data/processed/hateful_test.json; do
    if [[ ! -f "${required}" ]]; then
        echo "[ERROR] Missing ${required}" >&2
        exit 1
    fi
done

${PYTHON_BIN} -m src.acm_revision.prepare_data --config "${CONFIG}"

echo
if [[ -f "${REPORT}" ]]; then
    echo "================ split_report.json ========================="
    cat "${REPORT}"
    echo
    echo "============================================================"
    echo "[OK] Strict Hateful Memes pools were generated successfully."
    echo "[OK] Original train examples were preserved; duplicate families were grouped, not deleted."
    echo "Output: data/processed/acm_revision/hateful_memes/"
else
    echo "[ERROR] Command finished but ${REPORT} was not created." >&2
    exit 1
fi
