#!/usr/bin/env bash
set -euo pipefail

if [[ ! -d "src" || ! -d "configs" ]]; then
  echo "[ERROR] Run this script from the MFL-new project root."
  exit 1
fi

mkdir -p logs

BASIS="results/defense_seed42_assoc07/subspace/final_r5.npz"
SHADOW="results/defense_seed42_assoc07/subspace/combined/associated_all.npz"

[[ -f "$BASIS" ]] || { echo "[ERROR] Missing $BASIS"; exit 1; }
[[ -f "$SHADOW" ]] || { echo "[ERROR] Missing $SHADOW"; exit 1; }

echo "========================================"
echo "Running rank=5, alpha=1.0, noise=0.05 on GPU 1"
echo "Log: logs/defense_r5_a10_noise005_gpu1.log"
echo "========================================"

CUDA_VISIBLE_DEVICES=1 python -u -m src.scripts.run_defense_comparison   --method proposed   --basis "$BASIS"   --shadow-updates "$SHADOW"   --rank 5   --alpha 1.0   --gaussian-noise-ratio 0.05   2>&1 | tee logs/defense_r5_a10_noise005_gpu1.log

echo "========================================"
echo "Completed rank=5, alpha=1.0, noise=0.05 on GPU 1"
echo "========================================"
