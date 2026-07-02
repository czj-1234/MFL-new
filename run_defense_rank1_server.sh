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

echo "Starting rank=1, alpha=0.5 on GPU 0"
CUDA_VISIBLE_DEVICES=0 nohup python -u -m src.scripts.run_defense_comparison \
  --method all \
  --basis "$BASIS" \
  --shadow-updates "$SHADOW" \
  --rank 1 \
  --alpha 0.5 \
  > logs/defense_r1_server.log 2>&1 &

PID=$!
echo "$PID" > logs/defense_r1_server.pid
echo "Started PID=$PID"
echo "Monitor: tail -f logs/defense_r1_server.log"
