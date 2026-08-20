# Hateful Memes privacy-150 protocol

This is the authoritative protocol for the five-seed, 150-round modality-exclusive privacy experiment.
It replaces the earlier utility-only `run_modality_exclusive_curve.sh` workflow.

## Fixed data and FL settings

- Strict pools remain unchanged: shadow-train/shadow-val/target = 40/20/40.
- Seeds: 42, 43, 44, 45, 46.
- Concentrations: 0.5, 0.6, 0.7, 0.8, 0.9, 1.0.
- Populations: shadow_train, shadow_val, target.
- 12 clients, full participation, FedAvg, 150 rounds, one local epoch, batch size 16.
- shadow_train and target: 200 unique examples/client (2400 total).
- shadow_val: 100 unique examples/client (1200 total).
- No sample reuse across clients and no cross-pool overlap.

Total formal FL jobs: 5 seeds x 6 concentrations x 3 populations = 90.

## Update capture

Exact every round:

- classifier_bias
- classifier_weight
- classifier_head

Milestone rounds: 1, 5, 10, 20, 30, 50, 75, 100, 125, 150.

Exact at milestones:

- fusion
- missing_modality

Deterministic coordinate sketch at milestones (default 16,384 coordinates):

- image_encoder
- text_encoder
- all_shared
- full_update

Large groups are sketched at capture time to keep the experiment operational. The original dimension,
representation type, sketch dimension, and fixed sketch seed are written into every NPZ capture manifest.
Do not describe these four representations as exact full vectors in the paper; describe them as fixed,
deterministic coordinate sketches computed from the corresponding complete update group.

Every completed run must contain:

- `summary.json`
- `round_metrics.csv`
- `update_metadata.csv`
- `client_partition_manifest.json`
- `capture_manifest.json` with `status: PASS`
- non-empty `updates/*.npz`

A job is skipped only when both the requested 150-round summary and a PASS capture manifest exist.
Partial jobs are deleted and restarted in their isolated job directory.

## Privacy outputs

For each concentration, post-processing combines all five seeds and all three strict populations and writes:

- `attack_summary__<group>.csv`
- `canonical_leakage_summary.csv`
- `leakage_probabilities__<group>.csv`
- `label_distribution_summary.csv`
- `label_distribution_predictions__<group>.csv`
- `structural_privacy.csv`
- `trajectory_attacks.csv`
- `postprocess_report.json` with `status: PASS`

The structural file includes update norms, effective rank, top-k SVD energy, within/between-label cosine
similarity, cosine separation gap, Silhouette score, Davies-Bouldin index, and Calinski-Harabasz index.

The canonical leakage probability is produced by logistic regression trained on shadow_train, with C selected
only on shadow_val, and evaluated only on target. Per-target-update class probabilities are retained.

## Mandatory smoke test

Before a server starts its formal queue, it runs a two-round smoke test over shadow_train, shadow_val, and
target. The smoke test uses four clients, eight samples/client, one local step, and a 256-dimensional sketch.
The formal queue starts only after capture validation and post-processing produce a PASS marker.

## Three-GPU allocation

Server A GPU 0: concentrations 0.5 and 0.8.

Server A GPU 1: concentrations 0.6 and 0.9.

Server B GPU 1: concentrations 0.7 and 1.0.

Each GPU processes both assigned concentrations, all five seeds, and all three populations.
