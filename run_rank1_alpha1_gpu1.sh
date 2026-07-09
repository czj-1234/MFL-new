#!/usr/bin/env bash
set -euo pipefail

if [[ ! -d "src" || ! -d "configs" ]]; then
  echo "[ERROR] Run this script from the MFL-new project root."
  exit 1
fi

mkdir -p logs

SHADOW="results/defense_seed42_assoc07/subspace/combined/associated_all.npz"
ATTACK_BASIS="results/defense_seed42_assoc07/subspace/final_attack_aware_r5.npz"

[[ -f "$SHADOW" ]] || { echo "[ERROR] Missing $SHADOW"; exit 1; }
[[ -f "$ATTACK_BASIS" ]] || { echo "[ERROR] Missing $ATTACK_BASIS"; exit 1; }

echo "========================================"
echo "Running attack-aware rank=1, alpha=1.0 on GPU 1"
echo "Log: logs/defense_attack_aware_r1_a10_gpu1.log"
echo "========================================"

CUDA_VISIBLE_DEVICES=1 python -u -m src.scripts.run_defense_comparison \
  --method proposed \
  --basis "$ATTACK_BASIS" \
  --shadow-updates "$SHADOW" \
  --rank 1 \
  --alpha 1.0 \
  --output-root results/defense_seed42_assoc07/end_to_end_attack_aware \
  2>&1 | tee logs/defense_attack_aware_r1_a10_gpu1.log

echo "========================================"
echo "Completed attack-aware rank=1, alpha=1.0 on GPU 1"
echo "========================================"
