#!/usr/bin/env bash
set -euo pipefail

if [[ ! -d "src" || ! -d "configs" ]]; then
  echo "[ERROR] Run this script from the MFL-new project root."
  exit 1
fi

mkdir -p logs

BASIS="results/defense_seed42_assoc07/subspace/final_r5.npz"

[[ -f "$BASIS" ]] || { echo "[ERROR] Missing $BASIS"; exit 1; }

echo "========================================"
echo "Running PCA baseline: rank=5, alpha=1.0 on GPU 0"
echo "Basis: $BASIS"
echo "Log: logs/defense_pca_r5_a10_gpu0.log"
echo "========================================"

CUDA_VISIBLE_DEVICES=0 python -u -m src.scripts.run_defense_comparison \
  --method pca \
  --basis "$BASIS" \
  --rank 5 \
  --alpha 1.0 \
  2>&1 | tee logs/defense_pca_r5_a10_gpu0.log

echo "========================================"
echo "Completed PCA baseline: rank=5, alpha=1.0 on GPU 0"
echo "========================================"
