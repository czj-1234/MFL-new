# Minimal classifier-head repair (2 servers / 4 GPUs)

Branch: `acm-headfix-minimal`

This protocol changes only the classifier-head sensitive basis. The existing strong
Defense70 bases are reused for fusion, missing-modality, image, and text groups.

Formal matrix: 8 runs total

- shadow_train: 142, 143
- shadow_val: 242
- target: 42, 43, 44, 45, 46

The queues contain exactly two runs per physical GPU.

## 1. Update both servers

```bash
cd /data/deli/MFL-new/MFL-new
git fetch origin
git checkout acm-headfix-minimal
git pull origin acm-headfix-minimal
conda activate mfl
```

## 2. Prepare the signed classifier-head basis on Server A only

This reuses the existing Defense70 seed-142 reference checkpoints.

```bash
cd /data/deli/MFL-new/MFL-new
conda activate mfl
CUDA_VISIBLE_DEVICES=0 bash scripts/acm_revision/prepare_headfix.sh
```

Inspect the locked validation result before launching the eight formal runs:

```bash
cat results/acm_revision/headfix_prep/locked_headfix.yaml
```

The key field is `selected.shadow_val_oriented_auroc`. If it remains near 1.0,
stop and inspect the basis before spending time on formal target runs.

## 3. Copy the locked assets and generated configs to Server B

Replace `SERVER_B` with the actual SSH host.

```bash
rsync -av results/acm_revision/headfix_prep/ \
  SERVER_B:/data/deli/MFL-new/MFL-new/results/acm_revision/headfix_prep/

rsync -av results/acm_revision/defense70_prep/bases/ \
  SERVER_B:/data/deli/MFL-new/MFL-new/results/acm_revision/defense70_prep/bases/

rsync -av results/acm_revision/defense70_prep/locked_params.yaml \
  SERVER_B:/data/deli/MFL-new/MFL-new/results/acm_revision/defense70_prep/locked_params.yaml

rsync -av configs/acm_revision/generated/headfix/ \
  SERVER_B:/data/deli/MFL-new/MFL-new/configs/acm_revision/generated/headfix/
```

If both servers share the same filesystem, this copy step is unnecessary.

## 4. Launch four queues

Server A:

```bash
cd /data/deli/MFL-new/MFL-new
conda activate mfl
mkdir -p logs/headfix

nohup bash scripts/acm_revision/run_headfix_queue.sh \
  configs/acm_revision/generated/headfix/serverA_gpu0.tsv 0 \
  > logs/headfix/serverA_gpu0.queue.log 2>&1 &

nohup bash scripts/acm_revision/run_headfix_queue.sh \
  configs/acm_revision/generated/headfix/serverA_gpu1.tsv 1 \
  > logs/headfix/serverA_gpu1.queue.log 2>&1 &
```

Server B:

```bash
cd /data/deli/MFL-new/MFL-new
conda activate mfl
mkdir -p logs/headfix

nohup bash scripts/acm_revision/run_headfix_queue.sh \
  configs/acm_revision/generated/headfix/serverB_gpu0.tsv 0 \
  > logs/headfix/serverB_gpu0.queue.log 2>&1 &

nohup bash scripts/acm_revision/run_headfix_queue.sh \
  configs/acm_revision/generated/headfix/serverB_gpu1.tsv 1 \
  > logs/headfix/serverB_gpu1.queue.log 2>&1 &
```

Check progress:

```bash
nvidia-smi
tail -f logs/headfix/serverA_gpu0.queue.log
```

Each runner uses the resume-capable FL implementation, with a round checkpoint every
five rounds.

## 5. Gather Server B results onto Server A

```bash
rsync -av \
  SERVER_B:/data/deli/MFL-new/MFL-new/results/acm_revision/headfix_r150/ \
  /data/deli/MFL-new/MFL-new/results/acm_revision/headfix_r150/
```

## 6. Run the same strict post-processing protocol as Core72

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

The new capture protocol is intentionally aligned to the final Core72 reference:
classifier head exact every round, milestone rounds
`1,5,10,20,30,50,75,100,125,150`, 16,384-dimensional encoder/all-shared
coordinate sketches, and a 2,048-dimensional full-update signed feature-hash
projection.
