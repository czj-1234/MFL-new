# ACM Transactions Revision Experiment Suite

Branch: `acm-transactions-revision`

This branch is an isolated reimplementation of the paper's formal experiments. The old experiments on `main` remain useful as pilot/permissive diagnostics, but the claims in the revised paper should be based on this strict suite.

## Non-negotiable protocol rules

1. `shadow_train`, `shadow_val`, and `target` are disjoint raw-example pools. The official task validation/test data are never used to create attack/shadow clients.
2. Raw examples are never reused across clients inside a strict FL population.
3. `shadow_train` trains attacks and estimates bases. `shadow_val` tunes attack/defense hyperparameters. `target` is used once for final evaluation.
4. Every reported central result uses genuinely independent FL runs, not repeated attacker seeds on one FL trajectory.
5. Client/trajectory/run is the statistical unit; client-round updates are not treated as independent experimental units.
6. The same attack preprocessing/tuning procedure is used across image-only, text-only and modality-exclusive regimes.
7. The 50/50 condition is an **attribution control**, not a deployable privacy defense.
8. The identical-checkpoint contrast is described as a **label-composition-associated matched contrast**, not a causal subspace unless a stronger causal design is separately justified.
9. Defense selection is performed on `shadow_val` only. No target attack/utility result may be used to choose rank, attenuation or baseline hyperparameters.
10. The strongest defense claim must be based on defense-aware **full-update** attacks on unseen target clients.

## Datasets

- Hateful Memes: binary classification, primary benchmark.
- MVSA-Multiple: original 3-class sentiment setting, independent multiclass generalization benchmark.

Base configs:

```text
configs/acm_revision/hateful_memes.yaml
configs/acm_revision/mvsa_multiple.yaml
```

The full experiment matrix is:

```text
configs/acm_revision/experiment_matrix.yaml
```

## Step 0 — Audit and create strict pools

Audit the original processed files:

```bash
python -m src.acm_revision.analysis_cli audit-data \
  --train data/processed/hateful_train.json \
  --val data/processed/hateful_val.json \
  --test data/processed/hateful_test.json \
  --output artifacts/acm_revision/hateful_memes/data_audit.json

python -m src.acm_revision.analysis_cli audit-data \
  --train data/processed/mvsa_3class_train.json \
  --val data/processed/mvsa_3class_val.json \
  --test data/processed/mvsa_3class_test.json \
  --output artifacts/acm_revision/mvsa_multiple/data_audit.json
```

The audit reports whether a semantically valid natural client grouping field (user/source/author/topic/etc.) actually exists. Do not invent a pseudo-natural partition if neither selected dataset contains one.

Create disjoint shadow/validation/target pools:

```bash
python -m src.acm_revision.cli prepare-data --config configs/acm_revision/hateful_memes.yaml
python -m src.acm_revision.cli prepare-data --config configs/acm_revision/mvsa_multiple.yaml
```

Each dataset receives a `split_report.json` with deduplication and overlap checks.

---

# E1 — Task Performance and FL Validity

Purpose: verify that every modality regime is a usable FL task before making privacy claims.

Regimes:

- image-only
- text-only
- modality-exclusive
- complete/full multimodal

Concentrations: 50/50, 60/40, 70/30, 80/20, 90/10.

Metrics: ACC, Macro-F1, AUROC, loss. Ten independent FL runs.

Generate/run:

```bash
python -m src.acm_revision.cli generate-jobs \
  --matrix configs/acm_revision/experiment_matrix.yaml \
  --profile E1_task_performance

bash scripts/acm_revision/run_profile.sh E1_task_performance 0,1
```

---

# E2 — Strict Dominant-Label Leakage

Purpose: establish whether dominant-label leakage survives strict evaluation.

The formal attack protocols are implemented in `src/acm_revision/attacks.py`:

- `random_update` — permissive diagnostic only
- `temporal` — early rounds train, late rounds test
- `cross_partition` — shadow population train, unseen target population test
- `cross_partition_temporal` — early shadow rounds train, late unseen target rounds test
- `cross_client` — completely unseen clients at attack test
- trajectory-level evaluation — one client sequence is one attack unit

Generate all FL populations:

```bash
python -m src.acm_revision.cli generate-jobs \
  --matrix configs/acm_revision/experiment_matrix.yaml \
  --profile E2_strict_leakage
```

After the shadow-train, shadow-val and target runs are complete, pass the matching `update_metadata.csv` files together:

```bash
python -m src.acm_revision.cli run-attacks \
  --metadata <shadow_train_metadata.csv> <shadow_val_metadata.csv> <target_metadata.csv> \
  --protocols random_update,temporal,cross_client,cross_partition,cross_partition_temporal \
  --groups classifier_head \
  --models random,majority,update_norm,cosine_centroid,logistic_regression,linear_svm,random_forest,boosted_trees,xgboost,mlp \
  --targets dominant_label \
  --output results/acm_revision/attacks/E2.csv
```

The revised paper should treat `cross_partition_temporal` and unseen-client results as primary evidence, not the random update split.

---

# E3 — Modality Amplification

Purpose: test whether modality-exclusive FL is genuinely more vulnerable than matched image-only and text-only FL under the **same strict protocol**.

Regimes:

- image-only
- text-only
- modality-exclusive
- complete multimodal

Concentrations:

- 50/50
- 60/40
- 70/30
- 80/20
- 90/10
- 100/0 extreme control

Generate:

```bash
python -m src.acm_revision.cli generate-jobs \
  --matrix configs/acm_revision/experiment_matrix.yaml \
  --profile E3_modality_amplification
```

Only retain the word **amplification** if modality-exclusive remains significantly more vulnerable under paired strict comparisons.

Paired comparison example:

```bash
python -m src.acm_revision.analysis_cli paired-test \
  --csv <run_level_attack_results.csv> \
  --value attack_asr \
  --left modality_exclusive \
  --right image_only \
  --output results/acm_revision/stats/modality_vs_image.json
```

---

# E4 — Independent Units, Client Scale and Partial Participation

Purpose: remove the four-client/pseudoreplication limitation.

Client scales:

- 20
- 50
- 100

Participation:

- 10%
- 20%
- 50%
- 100%

Ten independent FL seeds are used throughout.

```bash
python -m src.acm_revision.cli generate-jobs \
  --matrix configs/acm_revision/experiment_matrix.yaml \
  --profile E4_scale_participation
```

Report run-level and client/trajectory-level confidence intervals. Do not report confidence intervals that treat client-round updates as independent.

---

# E5 — Confounder Identification

Purpose: determine what the attacker is actually learning.

For the same saved update observations, predict:

1. dominant label
2. modality
3. persistent client identity
4. communication-stage/round group
5. arbitrary balanced random grouping

```bash
python -m src.acm_revision.cli run-attacks \
  --metadata <shadow_train> <shadow_val> <target> \
  --protocols random_update,temporal,cross_client,cross_partition,cross_partition_temporal \
  --groups classifier_head,full_update \
  --models logistic_regression,random_forest,xgboost,mlp \
  --targets dominant_label,modality,client_id,round,random_group \
  --output results/acm_revision/attacks/E5_confounders.csv
```

Client-identity prediction is meaningful in permissive/temporal diagnostics; under a client-disjoint split the test client IDs are intentionally unseen, so that task is not interpreted as a standard closed-set classifier result.

---

# E6 — Strong Attack Family, Label Distribution and Trajectories

## Attack hyperparameter tuning

Tune attacks on attack-train/attack-validation only, then freeze the selected configuration before the target test:

```bash
python -m src.acm_revision.analysis_cli tune-attack \
  --metadata <shadow_train> <shadow_val> <target> \
  --protocol cross_partition_temporal \
  --group classifier_head \
  --model mlp \
  --output artifacts/acm_revision/attack_tuning/mlp.json
```

Repeat the same tuning budget/procedure for every modality regime.

## Label-distribution inference baseline

```bash
python -m src.acm_revision.cli label-distribution \
  --metadata <shadow_train> <shadow_val> <target> \
  --protocol cross_partition_temporal \
  --group classifier_head \
  --num-classes 2 \
  --output results/acm_revision/attacks/hateful_distribution_argmax.csv
```

Use `--num-classes 3` for MVSA-Multiple. Compare argmax of the inferred distribution with direct dominant-label inference.

## Trajectory attacks

```bash
python -m src.acm_revision.cli trajectory \
  --metadata <shadow_train> <shadow_val> <target> \
  --protocol cross_partition \
  --group classifier_head \
  --methods temporal_mean,temporal_variance,mean_variance,concatenation,lstm,tcn,transformer \
  --sequence-length 10 \
  --output results/acm_revision/attacks/E6_trajectory.csv
```

Client-level majority vote over single-round predictions:

```bash
python -m src.acm_revision.analysis_cli trajectory-vote \
  --metadata <shadow_train> <shadow_val> <target> \
  --protocol cross_partition \
  --group classifier_head \
  --output results/acm_revision/attacks/E6_vote.json
```

## Informed honest-but-curious server

The informed attack can append round, sample-size, participation, modality/architecture/aggregation metadata and global-checkpoint norm statistics to update features:

```bash
python -m src.acm_revision.analysis_cli informed-attack \
  --metadata <shadow_train> <shadow_val> <target> \
  --protocol cross_partition_temporal \
  --group full_update \
  --model mlp \
  --output results/acm_revision/attacks/E6_informed.json
```

Do not include persistent client identity in a client-disjoint test unless the threat model explicitly defines how unseen identities are represented.

---

# E7 — Layer-Wise and Full-Update Leakage

Attack observations separately from:

- classifier bias
- classifier weight
- full classifier head
- fusion/projector
- image encoder
- text encoder
- missing-modality parameters
- all shared layers
- complete model update

```bash
python -m src.acm_revision.cli run-attacks \
  --metadata <shadow_train> <shadow_val> <target> \
  --protocols cross_partition_temporal,cross_client \
  --groups classifier_bias,classifier_weight,classifier_head,fusion,image_encoder,text_encoder,missing_modality,all_shared,full_update \
  --models logistic_regression,random_forest,xgboost,mlp \
  --targets dominant_label \
  --output results/acm_revision/attacks/E7_layers.csv
```

This experiment determines whether a head-only defense is scientifically defensible or whether the sensitive signal is simply available elsewhere.

---

# E8 — Leakage Mechanism and Ablations

The matrix includes one-factor ablations for:

- fusion: concat, product, absolute difference, concat+product+absdiff
- early vs late fusion
- missing-modality handling: learned token, zero, fixed token, separate heads
- encoder training: frozen, partially trainable, fully fine-tuned

```bash
python -m src.acm_revision.cli generate-jobs \
  --matrix configs/acm_revision/experiment_matrix.yaml \
  --profile E8_mechanism_ablations
```

Geometry/mechanism outputs:

```bash
python -m src.acm_revision.cli mechanism \
  --metadata <metadata...> \
  --groups classifier_bias,classifier_weight,classifier_head,fusion,image_encoder,text_encoder,all_shared,full_update \
  --output results/acm_revision/mechanism/layer_geometry.csv
```

Implemented diagnostics include:

- update L2 norm
- same/different dominant-role cosine similarity
- within-role and between-role distances
- separation gap
- singular-value spectrum
- effective rank
- top-k energy
- image-text update/gradient alignment
- fixed-checkpoint class-conditional gradient direction probes

Clustering should only be interpreted on held-out clients/populations; do not map four persistent training clients to privacy conclusions.

---

# E9 — Identical-Checkpoint Matched Contrast

The old design that subtracts updates from two independently evolving global FL trajectories is not the formal method in this branch.

First run a reference `shadow_train` FL trajectory with every global checkpoint saved:

```bash
python -m src.acm_revision.cli generate-jobs \
  --matrix configs/acm_revision/experiment_matrix.yaml \
  --profile E9_identical_checkpoint_contrast
```

At the **same checkpoint**, branch the same shadow client into concentrated and balanced local-update calculations. The code keeps the global model state, client identity, modality, sample budget, optimizer configuration, LR, local epochs/steps and shuffle seed matched, while changing the local label composition.

Fit a basis:

```bash
python -m src.acm_revision.cli fit-basis \
  --config <reference_config.yaml> \
  --reference-run-dir <reference_run_dir> \
  --pool-json data/processed/acm_revision/hateful_memes/shadow_train.json \
  --contrast-dir artifacts/acm_revision/hateful_memes/contrasts \
  --rounds 1,5,10,20,30,40,50 \
  --groups classifier_head,fusion,image_encoder,text_encoder,all_shared,full_update \
  --concentrated 0.7 \
  --balanced 0.5 \
  --output artifacts/acm_revision/hateful_memes/contrast_basis.npz
```

For MVSA-Multiple use `--balanced 0.3333333333`.

The output is an ordered basis. **Rank is not selected here**; rank/attenuation are selected later on independent `shadow_val` results.

`basis_stability()` in `src/acm_revision/contrast.py` computes principal-angle/subspace-cosine stability across seeds, populations, datasets (when dimensions match) and architectures.

---

# E10 — Defense Baselines and Privacy–Utility Curves

## Baselines

Implemented update defenses:

- gradient clipping
- clipping + Gaussian noise
- local/client-update DP-style clipping + Gaussian noise, with RDP accounting helper
- quantization
- random sparsification
- random projection
- top-k pruning
- supervised sensitive-direction removal
- gradient orthogonalization
- rank-matched random subspace
- rank-matched PCA
- proposed contrast-aware filtering
- head-only, all-shared and full-update variants
- separate-basis/separate-alpha layer-wise adaptive filtering

Secure aggregation is handled separately because it removes individual pre-aggregation updates from the server observation model rather than sanitizing one client update.

Generate basis-free baselines:

```bash
python -m src.acm_revision.cli generate-jobs \
  --matrix configs/acm_revision/experiment_matrix.yaml \
  --profile E10A_defense_baselines
```

Fit rank-matched bases **only from shadow-train updates**:

```bash
python -m src.acm_revision.cli fit-baseline-basis \
  --method pca \
  --metadata <shadow_train_metadata...> \
  --group classifier_head \
  --rank 4 \
  --output artifacts/acm_revision/hateful_memes/pca_classifier_head.npz

python -m src.acm_revision.cli fit-baseline-basis \
  --method supervised \
  --metadata <shadow_train_metadata...> \
  --group classifier_head \
  --rank 4 \
  --output artifacts/acm_revision/hateful_memes/supervised_classifier_head.npz

python -m src.acm_revision.cli fit-baseline-basis \
  --method orthogonal \
  --metadata <shadow_train_metadata...> \
  --group classifier_head \
  --rank 4 \
  --output artifacts/acm_revision/hateful_memes/orthogonal_classifier_head.npz

python -m src.acm_revision.cli fit-baseline-basis \
  --method random \
  --metadata <shadow_train_metadata...> \
  --group classifier_head \
  --rank 4 \
  --output artifacts/acm_revision/hateful_memes/random_classifier_head.npz
```

Equivalent profiles exist for MVSA in the experiment matrix.

Generate the complete proposed privacy-utility sweep:

```bash
python -m src.acm_revision.analysis_cli defense-sweep \
  --base-config configs/acm_revision/hateful_memes.yaml \
  --basis artifacts/acm_revision/hateful_memes/contrast_basis.npz \
  --output-dir configs/acm_revision/generated/E10_hateful_contrast_sweep
```

Generate layer-wise adaptive attenuation:

```bash
python -m src.acm_revision.analysis_cli layerwise-sweep \
  --base-config configs/acm_revision/hateful_memes.yaml \
  --basis artifacts/acm_revision/hateful_memes/contrast_basis.npz \
  --output-dir configs/acm_revision/generated/E10_hateful_layerwise
```

The paper should plot attack ASR against task AUROC, Macro-F1 and ACC over the full sweep rather than report one selected point.

## Independent defense selection

Join shadow-validation utility/attack results into a table and select a point without target data:

```bash
python -m src.acm_revision.cli select-defense \
  --validation-results <shadow_validation_privacy_utility.csv> \
  --attack-metric attack_asr \
  --utility-metric task_auroc \
  --max-utility-drop 0.02 \
  --output artifacts/acm_revision/selected_defense.json
```

## DP accounting

```bash
python -m src.acm_revision.cli dp-epsilon \
  --noise-multiplier 1.0 \
  --sample-rate 0.2 \
  --steps 50 \
  --delta 1e-5
```

The paper must state the mechanism and accounting assumptions; do not describe clipping+noise as a formal DP guarantee without the corresponding accountant/assumptions.

---

# E11 — Adaptive Full-Update Defense and Generalization

## Adaptive/defense-aware attack

A valid adaptive evaluation trains the attacker on **defended shadow updates**, not raw updates:

```text
Defended shadow_train FL
        ↓
observed defended updates
        ↓
train attacker
        ↓
Defended unseen target FL
        ↓
full-update attack
```

The existing strict attack code already does this when `--observation observed` is supplied with defended shadow and defended target metadata.

The primary defense endpoint should be:

```text
unseen target clients + cross-partition/temporal split + defense-aware attacker + full update
```

Also compare head/fusion/encoder/full-update ASR to verify that filtering did not merely move the sensitive signal to another layer.

## Architecture generalization

```bash
python -m src.acm_revision.cli generate-jobs \
  --matrix configs/acm_revision/experiment_matrix.yaml \
  --profile E11A_architecture_generalization
```

Architectures:

1. CLIP dual encoder
2. ResNet18 + RoBERTa + fusion module

## Optimization generalization

```bash
python -m src.acm_revision.cli generate-jobs \
  --matrix configs/acm_revision/experiment_matrix.yaml \
  --profile E11B_optimization_generalization
```

Varies:

- FedAvg / FedProx
- AdamW / SGD
- local epochs 1 / 2 / 5
- batch size 16 / 32 / 64

## System generalization

```bash
python -m src.acm_revision.cli generate-jobs \
  --matrix configs/acm_revision/experiment_matrix.yaml \
  --profile E11C_system_generalization
```

Varies:

- 20 / 50 / 100 clients
- 10% / 20% / 50% / 100% participation
- FedAvg / FedProx

## Client-data-size robustness

```bash
python -m src.acm_revision.cli generate-jobs \
  --matrix configs/acm_revision/experiment_matrix.yaml \
  --profile E11D_client_data_size
```

Varies 25 / 50 / 75 / 100 examples per client at fixed 20 clients.

## Secure aggregation comparison

```bash
python -m src.acm_revision.analysis_cli secure-aggregation \
  --metadata <target_metadata...> \
  --group full_update \
  --output-dir results/acm_revision/secure_aggregation
```

This deliberately produces aggregate-round observations and records that an individual dominant-label target is no longer directly defined. Do not report it as though it were another per-client filtering ASR.

---

# Statistics and Reporting

Run-level aggregation with 95% bootstrap intervals:

```bash
python -m src.acm_revision.analysis_cli aggregate-runs \
  --csv <run_level_results.csv> \
  --metrics attack_asr,attack_auroc,test_auroc,test_macro_f1,test_acc \
  --groups dataset,setting_name,concentration,protocol,attack_model \
  --output results/acm_revision/stats/run_summary.csv
```

Client/trajectory-level bootstrap:

```bash
python -m src.acm_revision.analysis_cli bootstrap \
  --csv <client_level_results.csv> \
  --value attack_asr \
  --unit client_key \
  --output results/acm_revision/stats/client_bootstrap.json
```

Capture environment/software/GPU information:

```bash
python -m src.acm_revision.analysis_cli environment \
  --output artifacts/acm_revision/environment.json
```

Each final paper table should state what the mean, standard deviation and confidence interval are taken over.

---

# Large-Scale Execution

Single job:

```bash
bash scripts/acm_revision/run_job.sh <config.yaml> 0
```

Local multi-GPU profile:

```bash
bash scripts/acm_revision/run_profile.sh E2_strict_leakage 0,1
```

SLURM:

1. Generate the profile first.
2. Count rows in `jobs.csv`.
3. Submit a zero-based array, for example:

```bash
PROFILE=E2_strict_leakage sbatch --array=0-899 scripts/acm_revision/slurm_array.sh
```

Use the actual final row count rather than copying the example range.

---

# Mapping to Paper RQs

| Paper question | Formal experiment evidence |
| --- | --- |
| RQ1: Does dominant-class leakage exist? | E2, E4, E6 |
| RQ2: Does modality-exclusive FL amplify it? | E3 + paired strict statistics |
| RQ3: What is the attacker learning and why? | E5, E7, E8, E9 |
| RQ4: Can it be mitigated reliably? | E10, E11 adaptive full-update evaluation |

The revised Abstract/Introduction/Contributions/Conclusion should only be rewritten after these strict results determine which claims survive.
