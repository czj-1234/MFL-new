#!/usr/bin/env bash
#SBATCH --job-name=acm-revision
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/acm_revision/slurm_%A_%a.out

set -euo pipefail

PROFILE="${PROFILE:?Set PROFILE, e.g. E2_strict_leakage}"
MANIFEST="${MANIFEST:-configs/acm_revision/generated/${PROFILE}/jobs.csv}"

# SLURM_ARRAY_TASK_ID is zero-based in the recommended submission command.
CONFIG=$(python - "$MANIFEST" "$SLURM_ARRAY_TASK_ID" <<'PY'
import pandas as pd
import sys
manifest, idx = sys.argv[1], int(sys.argv[2])
df = pd.read_csv(manifest)
if idx < 0 or idx >= len(df):
    raise SystemExit(f"array index {idx} outside 0..{len(df)-1}")
print(df.iloc[idx]["config_path"])
PY
)

python -m src.acm_revision.cli run-fl --config "$CONFIG"
