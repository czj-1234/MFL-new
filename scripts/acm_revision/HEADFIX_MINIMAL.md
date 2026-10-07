# Minimal classifier-head repair (2 servers / 4 GPUs)

Branch: `acm-headfix-minimal`

This branch now uses **HeadFix-Minimal-v2**.

The preparation stage is fully offline: it reuses the existing exact classifier-head
updates from Core72 and does **not** retrain CLIP or rerun FL. It learns an iterative
attacker-guided nullspace from shadow-train seeds 142/143 (rounds <= 60), validates
on shadow-val seed 242 (rounds >= 61), and never uses target runs for basis selection.

The formal matrix is generated only if the strongest shadow-validation attacker among
LR/RF/MLP falls below the launch threshold (default 0.80).

## 1. Update both servers

```bash
cd /data/deli/MFL-new/MFL-new
git fetch origin
git checkout acm-headfix-minimal
git pull origin acm-headfix-minimal
conda activate mfl
```

## 2. Run the offline head search on Server A

```bash
cd /data/deli/MFL-new/MFL-new
conda activate mfl
bash scripts/acm_revision/prepare_headfix.sh
```

`CUDA_VISIBLE_DEVICES=0` is harmless but unnecessary because this stage does not
perform CLIP training.

Progress is printed in the foreground:

```text
[HEADFIX 1/3] loading existing exact classifier-head updates ...
[HEADFIX 2/3] iterative attacker-guided nullspace search ...
[HEADFIX BASIS] learned direction 1/64
...
[HEADFIX 3/3] evaluating LR/RF/MLP on shadow validation
[HEADFIX EVAL] rank=... worstAUC=... strongest=... relChange=...
```

If the best selected worst-case validation AUROC is above 0.80, the script exits
without generating any formal queues. Do not run the expensive FL matrix in that case.

Inspect:

```bash
cat results/acm_revision/headfix_prep/locked_headfix.yaml
```

The key fields are:

```yaml
status:
selected:
  rank:
  alpha:
  shadow_val_worst_oriented_auroc:
  shadow_val_strongest_attack:
  shadow_val_mean_relative_head_change:
```

## 3. Formal matrix if status=LOCKED

Eight runs total:

- shadow_train: 142, 143
- shadow_val: 242
- target: 42, 43, 44, 45, 46

Queues are split two jobs per physical GPU:

- Server A GPU0: shadow_train 142, target 42
- Server A GPU1: shadow_train 143, shadow_val 242
- Server B GPU0: target 43, target 44
- Server B GPU1: target 45, target 46

If the two servers do not share storage, copy the newly generated head-fix assets and
configs to Server B before starting.

## 4. Foreground execution

Server A GPU0:

```bash
bash scripts/acm_revision/run_headfix_queue.sh \
  configs/acm_revision/generated/headfix/serverA_gpu0.tsv 0
```

Server A GPU1:

```bash
bash scripts/acm_revision/run_headfix_queue.sh \
  configs/acm_revision/generated/headfix/serverA_gpu1.tsv 1
```

Server B GPU0:

```bash
bash scripts/acm_revision/run_headfix_queue.sh \
  configs/acm_revision/generated/headfix/serverB_gpu0.tsv 0
```

Server B GPU1:

```bash
bash scripts/acm_revision/run_headfix_queue.sh \
  configs/acm_revision/generated/headfix/serverB_gpu1.tsv 1
```

The queues run in the foreground and also write per-job logs under `logs/headfix/`.
The FL runner saves a resume checkpoint every five communication rounds.

## 5. Post-process after gathering all eight runs

```bash
python -m src.acm_revision.core72_postprocess \
  --root results/acm_revision/headfix_r150 \
  --setting modality_exclusive \
  --concentration 0.7 \
  --rounds 150 \
  --output-root results/acm_revision/headfix_postprocess_r150 \
  --shadow-train-seeds 142,143 \
  --shadow-val-seeds 242 \
  --target-seeds 42,43,44,45,46
```

The formal HeadFix capture protocol is aligned to the final Core72 reference:
classifier head exact every round, milestone rounds
`1,5,10,20,30,50,75,100,125,150`, 16,384-dimensional encoder/all-shared
coordinate sketches, and a 2,048-dimensional full-update signed feature-hash
projection.
