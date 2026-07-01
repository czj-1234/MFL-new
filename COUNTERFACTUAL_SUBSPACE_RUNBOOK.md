# Counterfactual Subspace Defense Runbook

This branch implements the seed42, Hateful Memes, modality-exclusive, association=0.7 experiment plan.

Verified best seed42 training settings from `hateful_modality_exclusive_final42.zip`:

- rounds: 25
- local epochs: 1
- learning rate: 5e-6
- weight decay: 0.03
- FedProx mu: 0.001
- samples per client: 2000
- fixed partition with overlap enabled
- server calibration disabled
- checkpoint selection by validation AUROC

Reference best checkpoint: round 20, test accuracy 0.654, macro-F1 0.6521, test AUROC 0.7137.

## 0. Pull the branch

```bash
cd /data/deli/MFL-new/MFL-new
git fetch origin
git checkout counterfactual-subspace-defense
git pull origin counterfactual-subspace-defense
mkdir -p logs
```

## 1. Generate strict matched shadow pairs A/B/C

Each split uses the same modality-wise sample pool for its associated and counterfactual conditions. The associated clients use 70/30 label allocation, while the counterfactual clients use a balanced 50/50 redistribution of the same pool.

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u -m src.scripts.run_shadow_splits \
  --splits A B C \
  > logs/shadow_splits.log 2>&1 &
```

Monitor:

```bash
tail -f logs/shadow_splits.log
```

## 2. Learn one rank-5 basis per split

Rank 5 is learned once so later experiments can use its first 1, 3, or 5 directions.

```bash
mkdir -p results/defense_seed42_assoc07/subspace

for S in A B C; do
  python -m src.analysis.learn_counterfactual_subspace \
    --associated results/defense_seed42_assoc07/shadow/split_${S}/modality_exclusive/assoc_07/classifier_updates.npz \
    --counterfactual results/defense_seed42_assoc07/shadow/split_${S}/modality_exclusive/counterfactual/classifier_updates.npz \
    --rank 5 \
    --output results/defense_seed42_assoc07/subspace/split_${S}_r5.npz
done
```

## 3. Compare shadow subspace stability

```bash
python -m src.analysis.subspace_stability \
  --basis-a results/defense_seed42_assoc07/subspace/split_A_r5.npz \
  --basis-b results/defense_seed42_assoc07/subspace/split_B_r5.npz \
  --basis-c results/defense_seed42_assoc07/subspace/split_C_r5.npz \
  --output results/defense_seed42_assoc07/subspace/stability.json
```

## 4. Combine all shadow pairs and learn the final basis

```bash
python -m src.analysis.combine_shadow_updates \
  --associated \
    results/defense_seed42_assoc07/shadow/split_A/modality_exclusive/assoc_07/classifier_updates.npz \
    results/defense_seed42_assoc07/shadow/split_B/modality_exclusive/assoc_07/classifier_updates.npz \
    results/defense_seed42_assoc07/shadow/split_C/modality_exclusive/assoc_07/classifier_updates.npz \
  --counterfactual \
    results/defense_seed42_assoc07/shadow/split_A/modality_exclusive/counterfactual/classifier_updates.npz \
    results/defense_seed42_assoc07/shadow/split_B/modality_exclusive/counterfactual/classifier_updates.npz \
    results/defense_seed42_assoc07/shadow/split_C/modality_exclusive/counterfactual/classifier_updates.npz \
  --output-dir results/defense_seed42_assoc07/subspace/combined

python -m src.analysis.learn_counterfactual_subspace \
  --associated results/defense_seed42_assoc07/subspace/combined/associated_all.npz \
  --counterfactual results/defense_seed42_assoc07/subspace/combined/counterfactual_all.npz \
  --rank 5 \
  --output results/defense_seed42_assoc07/subspace/final_r5.npz
```

## 5. Run the target baseline and three end-to-end comparisons

First comparison: rank=3, alpha=0.5.

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u -m src.scripts.run_defense_comparison \
  --method all \
  --basis results/defense_seed42_assoc07/subspace/final_r5.npz \
  --shadow-updates results/defense_seed42_assoc07/subspace/combined/associated_all.npz \
  --rank 3 \
  --alpha 0.5 \
  > logs/defense_comparison.log 2>&1 &
```

This runs:

- baseline;
- random rank-3 removal;
- PCA rank-3 removal;
- proposed rank-3 removal.

## 6. Shadow-to-target transfer

```bash
python -m src.analysis.target_transfer \
  --target-updates results/defense_seed42_assoc07/end_to_end/baseline/modality_exclusive/0.7/classifier_updates.npz \
  --basis results/defense_seed42_assoc07/subspace/final_r5.npz \
  --output results/defense_seed42_assoc07/analysis/target_transfer.json
```

## 7. Offline keep/remove/random/PCA intervention

The attack models are retrained inside `compute_structure_metrics` for every transformed update matrix.

```bash
python -m src.analysis.leakage_intervention \
  --updates results/defense_seed42_assoc07/end_to_end/baseline/modality_exclusive/0.7/classifier_updates.npz \
  --basis results/defense_seed42_assoc07/subspace/final_r5.npz \
  --rank 3 \
  --alpha 1.0 \
  --output results/defense_seed42_assoc07/analysis/leakage_intervention_r3.json
```

## 8. Proposed parameter sweep

This runs alpha={0.25,0.5,0.75,1.0} at rank=3, then rank={1,3,5} at alpha=0.5. Duplicate configurations are skipped.

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u -m src.scripts.run_defense_sweep \
  --basis results/defense_seed42_assoc07/subspace/final_r5.npz \
  > logs/defense_sweep.log 2>&1 &
```

After reviewing the rank-3 alpha results, change `best_alpha` in `src/scripts/run_defense_sweep.py` if the best value is not 0.5, then rerun only the rank tests.

## 9. Summarize all full-training results

```bash
python -m src.analysis.summarize_defense \
  --root results/defense_seed42_assoc07 \
  --output results/defense_seed42_assoc07/defense_summary.csv
```

## Expected baseline reference

- Test Accuracy: 0.654
- Macro-F1: 0.6521
- Test AUROC: 0.7137
- K-means Accuracy: 1.0
- RF ASR: 1.0
- MLP ASR: 1.0

Initial success criterion: test accuracy at least 0.634, with proposed leakage reduction stronger than random and PCA at comparable utility loss.
