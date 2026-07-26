#!/usr/bin/env bash
set -euo pipefail

# Prepare strict Shadow-Train / Shadow-Val / Target pools for Hateful Memes.
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
echo "Repository : ${REPO_ROOT}"
echo "Config     : ${CONFIG}"
echo "Python     : $(${PYTHON_BIN} --version 2>&1)"
echo "============================================================"

if [[ ! -f "${CONFIG}" ]]; then
    echo "[ERROR] Missing config: ${CONFIG}" >&2
    echo "Make sure you are on branch: acm-transactions-revision" >&2
    exit 1
fi

if [[ ! -f "data/processed/hateful_train.json" ]]; then
    echo "[ERROR] Missing data/processed/hateful_train.json" >&2
    exit 1
fi
if [[ ! -f "data/processed/hateful_val.json" ]]; then
    echo "[ERROR] Missing data/processed/hateful_val.json" >&2
    exit 1
fi
if [[ ! -f "data/processed/hateful_test.json" ]]; then
    echo "[ERROR] Missing data/processed/hateful_test.json" >&2
    exit 1
fi

${PYTHON_BIN} -m src.acm_revision.cli prepare-data --config "${CONFIG}"

echo
if [[ -f "${REPORT}" ]]; then
    echo "================ split_report.json ========================="
    cat "${REPORT}"
    echo
    echo "============================================================"
    echo "[OK] Strict Hateful Memes pools were generated successfully."
    echo "Output: data/processed/acm_revision/hateful_memes/"
else
    echo "[ERROR] Command finished but ${REPORT} was not created." >&2
    exit 1
fi
