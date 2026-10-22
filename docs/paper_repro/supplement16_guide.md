# ASLWF supplemental 16 formal FL runs (two servers, four GPUs)

This is a **new branch** based on `paper-aslwf-repro-20261017`; old data, manifests and output folders are never overwritten. **No experiments have been executed merely by committing these scripts.**

## The two controlled conditions

| Server | Experiment | 150-round runs | GPU0 (4 jobs) | GPU1 (4 jobs) |
|---|---|---:|---|---|
| A | uncentered c=0.7 | 8 | shadow142, target42, target44, target46 | shadow143, shadow_val242, target43, target45 |
| B | frozen ASLWF c=0.9 | 8 | shadow142, target42, target44, target46 | shadow143, shadow_val242, target43, target45 |

Both have 12 fixed clients, 100% participation, AdamW 1e-6, batch 16, local epoch 1, 150 rounds, 8 milestones `{1,10,30,50,75,100,125,150}`, 16384-coordinate encoder/all-shared sketches, 2048-deterministic signed-hash **attack capture** for full update, and the same CP/CPT / nine-surface analysis protocol.

The frozen Defense70 **internal basis-lift hash** remains at the dimension and seed already recorded in `locked_params.yaml` (historically 512-d). It is separate from the 2048-d full-update **attack observation**; changing it would confound both ablations and invalidate comparison with historical Complete ASLWF. Do not alter the locked internal basis.

- **c=0.9:** reuses the complete centered ASLWF head, other group bases, rank, attenuation and L2 budgets selected at c=0.7. Re-trains an independent c=.9 defended shadow_train/shadow_val/target population. No re-tuning on target.
- **uncentered:** re-derives only the non-head matched-contrast bases from original *shadow-side identical checkpoint* contrasts **without mean subtraction**; retains the exact previously locked attacker-guided classifier-head basis, alpha/rank and L2 budgets. It first recomputes **centered bases from the identical contrasts** and validates principal-subspace parity with the original centered bases (min cosine ≥ .995) to reject confounded reproduction. Requires the original eight shadow reference checkpoints.
- Both methods apply defense client-side before the FedAvg server receives the transmitted update, preserving the original honest-but-curious observation model.
- Full original-space raw-update, filter-difference and relative L2 statistics for **each client upload** are added to `update_metadata.csv`, for the exact original head and full model, separately from 2048-d attack sketches.

## Prerequisites (on both servers)

Checkout the same branch `aslwf-supplement16-20261022`. Activate the existing Python/torch environment. The repository **must** contain actual locked manifests and bases:

- `results/acm_revision/defense70_prep/locked_params.yaml` and all listed basis NPZs;
- `results/acm_revision/headfix_prep/locked_headfix.yaml` and its basis NPZ;
- original Hateful Memes pools, pretrained model files, `configs/acm_revision/hateful_memes.yaml`.

Additionally **Server A** requires the original `configs/acm_revision/generated/defense70_prep/reference_seed142.yaml` and the actual eight reference checkpoints from `locked_params.yaml -> provenance.reference_run_dir`. It does **not** silently replace missing checkpoints with a newly trained trajectory. This protects centered/uncentered comparability.

If either server has only a copy of the GitHub repository without the actual original large manifests/checkpoints, the scripts stop with a clear missing-artifact error. Copy the actual frozen artifacts first and verify their origin; do not fabricate placeholder manifests.

## Commands

On Server A (two GPUs; four serial jobs on each):

```bash
git fetch origin
git checkout aslwf-supplement16-20261022
conda activate mfl
bash scripts/acm_revision/serverA_uncentered_8.sh
```

On Server B (start independently of A; two GPUs):

```bash
git fetch origin
git checkout aslwf-supplement16-20261022
conda activate mfl
bash scripts/acm_revision/serverB_c09_8.sh
```

If device IDs differ: `GPU0=1 GPU1=2 bash scripts/acm_revision/serverB_c09_8.sh`. Launch scripts once per server, not once per GPU. Each starts **two child GPU queues**. The pre-existing resumable `run_defense70_job.sh` handles GPU free-memory checks, checkpointing, retries, duplicate-pass skipping and an independent job completion audit. Re-run the same server script after an interruption to resume.

- Logs: `logs/acm_revision/supplement16/{uncentered,c09}/` (including per-GPU queue logs and per-job attempts).
- Runtime YAML: `logs/acm_revision/supplement16/<phase>/runtime_configs/`.
- Rolling resume checkpoints: `checkpoints/acm_revision/supplement16/<phase>/`.
- Results: `results/acm_revision/supplement16/{uncentered,c09}/`.
- Frozen job configs and GPU queues: `configs/acm_revision/generated/supplement16/<phase>/`.
- The eight runs per server include two shadow-training, one shadow-validation and **five independent target** runs (42–46). GPU queues do **not** share a mutable active model between jobs.

## Verified results and statistical postprocessing

The shell scripts automatically invoke:

```bash
python -m src.acm_revision.supplement16 verify --phase uncentered  # server A
python -m src.acm_revision.supplement16 verify --phase c09          # server B
```

These require 8 complete, successful 150-round runs, exact eight capture milestones and all 5 target seeds. They output per-target **exact original-space actual relative L2** (mean, median, p95) and paired-run-unit five-target task AUROC mean, SD, bootstrap 95% CI in `results/acm_revision/supplement16/audits/<phase>_statistics.json`.

Run the previously implemented defense-aware nine-surface CP/CPT LR/RF/MLP audit separately after finishing each condition and providing the **verified, matching no-defense baseline task AUROC CSV**:

```bash
python scripts/acm_revision/audit_headfix.py \
  --root results/acm_revision/supplement16/c09 \
  --out results/acm_revision/supplement16/audits/c09_attacks \
  --concentration 0.9 \
  --baseline-task-csv /path/to/verified_c09_nodefense_task.csv
```

For uncentered use `--root results/acm_revision/supplement16/uncentered` and `--concentration 0.7`. The audit selects observation surface, attack model and AUC score orientation exclusively on shadow validation. It then computes five target seed results and uncertainty; it **does not** use target labels to tune the attack. The reused original centered-ASLWF c=.7 main result should also be audited under the same frozen protocol before any paired comparison.

## Matched-update L2 audit for the *old* PCA/Random/ASLWF baselines

Because historical raw full-model updates might be absent, this branch includes a separate shadow-only **matched-update perturbation** procedure. Run on Server A after basis preparation and preferably after the 16 FL runs (to avoid competing for resources):

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.acm_revision.supplement16_distortion \
  --rounds 10,75,150 --clients 12
```

It replays **one identical raw update per checkpoint/client**, then passes that same update through the locked baseline defenses and Complete/Uncentered ASLWF. Output `results/acm_revision/supplement16/audits/matched_update/summary.csv` includes exact-model L2 mean/median/p95 and raw/difference norms for PCA/Random/ASLWF, with clip-Gaussian tagged separately (different mechanism). It uses **shadow-side auxiliary data only**. This is a checkpoint-matched comparative audit, **not** a substitute for per-step historical end-to-end actual distortion, and it produces no new attack/utility claims.

## Integrity constraints

1. Never merge output directories of different conditions and never overwrite old Complete-ASLWF run directories.
2. Do not change the old locked manifests, hash-lift ranks, alpha, seeds or step budgets to make parity pass.
3. If A fails the centered-parity test or a checkpoint is missing, stop and investigate rather than silently using alternate checkpoint trajectories or training new bases on target data.
4. If historical `c=.9` no-defense outputs differ in original encoding/training/capture protocol, do not claim direct comparability without revalidating.
5. A finished 16-run campaign alone does not establish new numerical results until the attacks and actual L2 audit complete. Report uncertainty over independent *target runs*, not thousands of correlated per-client/round updates.
