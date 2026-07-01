#!/bin/bash

# ============================================================
# Hateful Memes 2-class experiments - Server B
# Run modality_exclusive
# ============================================================

cd /data/deli/MFL-new/MFL-new || exit 1

SETTINGS=("modality_exclusive" "image_only" "text_only")
ASSOCIATIONS=("iid" "0.7" "0.9" "1.0")

mkdir -p logs

for setting in "${SETTINGS[@]}"; do
  for association in "${ASSOCIATIONS[@]}"; do

    echo ""
    echo "============================================================"
    echo "GPU 1 / B running: setting=${setting}, association=${association}"
    echo "============================================================"
    echo ""

    CUDA_VISIBLE_DEVICES=1 /home/deli/Data/miniconda3/envs/mfl/bin/python -m src.main \
      --config configs/config_hateful_modality_exclusive1.yaml \
      --setting "${setting}" \
      --association "${association}" \
      --rounds 25 \
      2>&1 | tee "logs/hateful_B_${setting}_${association}.log"

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
      echo ""
      echo "Experiment failed on Server B: setting=${setting}, association=${association}"
      exit 1
    fi

  done
done

echo ""
echo "Server B Hateful Memes experiments completed."
