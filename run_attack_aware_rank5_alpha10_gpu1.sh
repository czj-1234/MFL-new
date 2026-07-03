#!/usr/bin/env bash
set -euo pipefail

if [[ ! -d "src" || ! -d "configs" ]]; then
  echo "[ERROR] Run this script from the MFL-new project root."
  exit 1
fi

mkdir -p logs

SHADOW="results/defense_seed42_assoc07/subspace/combined/associated_all.npz"
CF_BASIS="results/defense_seed42_assoc07/subspace/final_r5.npz"
ATTACK_BASIS="results/defense_seed42_assoc07/subspace/final_attack_aware_r5.npz"

[[ -f "$SHADOW" ]] || { echo "[ERROR] Missing $SHADOW"; exit 1; }
[[ -f "$CF_BASIS" ]] || { echo "[ERROR] Missing $CF_BASIS"; exit 1; }

echo "========================================"
echo "Step 1: Build attack-aware rank-5 basis"
echo "========================================"

python -u -m src.scripts.build_attack_aware_basis \
  --shadow-updates "$SHADOW" \
  --counterfactual-basis "$CF_BASIS" \
  --rank 5 \
  --seed 42 \
  --output "$ATTACK_BASIS" \
  2>&1 | tee logs/build_attack_aware_r5.log

[[ -f "$ATTACK_BASIS" ]] || { echo "[ERROR] Failed to create $ATTACK_BASIS"; exit 1; }

echo "========================================"
echo "Step 2: Run attack-aware rank=5, alpha=1.0 on GPU 1"
echo "========================================"

CUDA_VISIBLE_DEVICES=1 python -u -m src.scripts.run_defense_comparison \
  --method proposed \
  --basis "$ATTACK_BASIS" \
  --shadow-updates "$SHADOW" \
  --rank 5 \
  --alpha 1.0 \
  --output-root results/defense_seed42_assoc07/end_to_end_attack_aware \
  2>&1 | tee logs/defense_attack_aware_r5_a10_gpu1.log

echo "========================================"
echo "Completed attack-aware rank=5, alpha=1.0"
echo "========================================"
