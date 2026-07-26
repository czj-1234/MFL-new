# Review Comment → Implemented Experiment Mapping

This checklist is intended to prevent selective revision. Every major scientific comment in the ACM Transactions assessment has a corresponding implementation in `src/acm_revision/`.

| Major comment | Required scientific change | Implementation |
| --- | --- | --- |
| 1. Strict modality-amplification protocol | Same cross-partition, temporal, cross-partition+temporal, cross-client and trajectory tests for every regime | `attacks.py`, `advanced_attacks.py`, `E2`, `E3` |
| 2. Too few independent units | 20/50/100 clients, partial participation, 10 genuinely independent FL seeds, client/run statistics | `fl.py`, `experiment_matrix.yaml` E4/E11, `analysis.py` |
| 3. Dominant-label novelty | Compare direct inference with full label-distribution inference→argmax | `run_label_distribution_inference()` in `attacks.py`, E6 |
| 4. Counterfactual attribution too strong | Same global checkpoint; matched local branches; associative terminology | `contrast.py`, E9 |
| 5. Shadow/target independence | Disjoint pools, no client sample reuse, exact/near duplicate audit, auxiliary-data sensitivity | `data_protocol.py`, `data_audit.py`, `auxiliary_sensitivity.py` |
| 6. Weak attacker family | LR, SVM, RF, boosting, XGBoost, MLP, norm/cosine, trajectories, LSTM/TCN/Transformer, informed server | `attacks.py`, `attack_tuning.py`, `advanced_attacks.py` |
| 7. Head-only defense | Bias/head/fusion/image encoder/text encoder/all shared/full update attack spaces; head/all-shared/full and layer-wise defenses | `defenses.py`, E7, E10, E11 |
| 8. Weak defense evaluation | Clipping/noise/DP-style/quantization/sparsification/random projection/top-k/supervised/orthogonal/random/PCA/proposed, independent tuning, Pareto sweeps | `defenses.py`, `basis_baselines.py`, `defense_sweep.py`, `baseline_sweep.py` |
| 9. Narrow experimental scope | Hateful Memes + MVSA-Multiple, binary+multiclass, CLIP + ResNet18/RoBERTa, client/participation/optimizer/aggregation variations | E1/E11, `architectures.py` |
| 10. Weak mechanism | Norm, cosine, within/between distance, update alignment, class-conditional gradients, SVD/effective rank, held-out clustering, fusion/missing/encoder ablations | `analysis.py`, `clustering.py`, E8 |
| 11. Threat/deployment assumptions | Informed server attack; explicit saved observation components; secure aggregation handled as a different observation model | `advanced_attacks.py`, `secure_aggregation.py`, metadata manifests |
| 12. Balanced condition is not a defense | 50/50 is used only as attribution/matched-composition control; deployable defenses modify updates while retaining concentrated target data | `contrast.py`, docs and E9/E10 separation |

## Additional reviewer-requested controls

### Auxiliary shadow-data sensitivity

The reviewer requested 10%, 25%, 50% and 100% auxiliary-data experiments. Run:

```bash
python -m src.acm_revision.supplemental_cli auxiliary-sensitivity \
  --metadata <shadow_train> <shadow_val> <target> \
  --protocol cross_partition_temporal \
  --group classifier_head \
  --model mlp \
  --fractions 0.10,0.25,0.50,1.00 \
  --output results/acm_revision/attacks/auxiliary_sensitivity.csv
```

The implementation subsamples **whole shadow-client trajectories**, not individual client-round updates.

### Held-out clustering without post-hoc role assignment

```bash
python -m src.acm_revision.supplemental_cli heldout-clustering \
  --metadata <heldout_target_metadata...> \
  --group classifier_head \
  --output results/acm_revision/mechanism/heldout_clustering.csv
```

It reports ARI/NMI against dominant label, modality, client identity, round group and a random grouping. This avoids claiming a cluster is a dominant-label cluster only after manually mapping it to a desired role.

### Rank-matched basis baselines across the same sweep

Fit PCA/random/supervised/orthogonal bases at the **largest** requested rank using shadow-train data, then generate lower-rank prefixes with the same rank/attenuation grid used by the proposed method:

```bash
python -m src.acm_revision.supplemental_cli basis-baseline-sweep \
  --base-config configs/acm_revision/hateful_memes.yaml \
  --pca artifacts/acm_revision/hateful_memes/pca_classifier_head.npz \
  --random artifacts/acm_revision/hateful_memes/random_classifier_head.npz \
  --supervised artifacts/acm_revision/hateful_memes/supervised_classifier_head.npz \
  --orthogonal artifacts/acm_revision/hateful_memes/orthogonal_classifier_head.npz \
  --group classifier_head \
  --ranks 1,2,4,8,16 \
  --alphas 0.25,0.5,0.75,1.0 \
  --output-dir configs/acm_revision/generated/E10_hateful_basis_sweep
```

### Natural client partition

`data_audit.py` reports candidate source/user/author/topic fields. A natural-client experiment should only be claimed if one of the chosen datasets actually contains a semantically defensible grouping. If neither Hateful Memes nor MVSA-Multiple provides such a field, document that fact instead of fabricating a “natural” partition from random groups.

### Cross-dataset transfer

The review requests domain-shift/cross-dataset transfer *if feasible*. Hateful Memes is binary while the selected MVSA-Multiple setup is 3-class, so a direct dominant-class attack transfer does not share the same target label space. Any cross-dataset transfer experiment must define a scientifically justified common target mapping before it is run; do not silently collapse the MVSA labels merely to create a transfer number.
