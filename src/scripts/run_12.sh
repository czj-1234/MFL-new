#!/bin/bash

# ============================================================
# Run 12 MVSA original 3-class full-data structural baseline experiments
#
# 3 settings × 4 association levels:
#   1. text_only
#   2. modality_exclusive
#   3. image_only
#
# For modality_exclusive:
#   - 3 clients
#   - client 0: text
#   - client 1: text
#   - client 2: image
#
# Make sure configs/config.yaml uses:
#   data.num_classes: 3
#   federated.num_clients: 3
#   data.train_json: data/processed/mvsa_3class_train.json
# ============================================================

cd /data/deli/MFL-new/MFL-new || exit 1

SETTINGS=("text_only" "modality_exclusive" "image_only")
ASSOCIATIONS=("iid" "0.3" "0.7" "1.0")

for setting in "${SETTINGS[@]}"; do
  for association in "${ASSOCIATIONS[@]}"; do

    echo ""
    echo "============================================================"
    echo "Running 3-class experiment: setting=${setting}, association=${association}"
    echo "============================================================"
    echo ""

    CUDA_VISIBLE_DEVICES=1 /home/deli/Data/miniconda3/envs/mfl/bin/python -m src.main \
      --setting "${setting}" \
      --association "${association}" \
      --rounds 100

    if [ $? -ne 0 ]; then
      echo ""
      echo "Experiment failed: setting=${setting}, association=${association}"
      exit 1
    fi

  done
done

echo ""
echo "All 12 MVSA original 3-class experiments completed."