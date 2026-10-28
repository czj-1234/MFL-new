# FINAL: 16 ASLWF federated training runs on two servers

This is the **only new branch** for the October 22 supplemental campaign:
`aslwf-final16-two-servers-20261022`. It preserves the paper-reproduction
foundation from `paper-aslwf-repro-20261017`, including the locked ASLWF
method; the old experimental branches are retired.

## Formal training count: 16, not 128

| Host | Scientific condition | Shadow train | Shadow validation | Target | FL total | GPUs |
|---|---|---|---|---|---:|---|
| Server A | c=.7 **uncentered** non-head matched contrasts, frozen attacker-guided classifier head | 142, 143 | 242 | 42–46 | **8** | 0 and 1 |
| Server B | c=.9 **frozen original Complete ASLWF from c=.7**, no c=.9 recalibration | 142, 143 | 242 | 42–46 | **8** | 0 and 1 |
| **Combined** | 2 supplemental conditions | 4 | 2 | 10 | **16** | **4 total** |

Each GPU handles **4 serial jobs**. The comparison for c=.7 centered ASLWF
comes from previously verified runs and is *not* silently retrained. This
campaign does not add MVSA, partial-participation, frozen-encoder, or
second-backbone FL retraining. The separate matched-update distortion audit
uses local probes at old checkpoints, **not** additional 150-round FL runs.

## Locked scientific protocol

- Dataset: Hateful Memes, binary, modality-exclusive, 12 clients.
- Model: CLIP ViT-L/14; all other architecture and optimization settings retained.
- 150 communication rounds; all 12 clients participate; one local epoch;
  AdamW learning rate 1e-6, batch size 16, FedAvg.
- Exactly eight sampling milestones:
  `1,10,30,50,75,100,125,150`.
- Original full-update **attacker representation** is a 2048-d signed hash;
  the **internal subspace basis lift** uses the dimension and seed in the
  locked `locked_params.yaml` (historically 512), **not** 2048.
- Nine attack observation surfaces and two CP/CPT evaluation protocols.
- Defense is applied per-client before upload; exact-space original-update
  L2 perturbation limits remain frozen and measured.

## Required *existing* real artifacts

Both machines require the same code revision, licensed/preprocessed Hateful
Memes strict pools, model cache, and the real locked method files:

```text
results/acm_revision/defense70_prep/locked_params.yaml
results/acm_revision/defense70_prep/bases/*.npz
results/acm_revision/headfix_prep/locked_headfix.yaml
results/acm_revision/headfix_prep/attacker_guided_classifier_head_basis.npz
```

Server A needs eight **original** seed142 shadow reference checkpoints
at the path specified in
`locked_params.yaml -> provenance.reference_run_dir`.
If only the generated YAML
`configs/acm_revision/generated/defense70_prep/reference_seed142.yaml`
is missing, Server A will **automatically reconstruct the YAML only** from
the original Defense70 generator and verify that its derived run path exactly
matches the existing locked checkpoint directory. It does **not** rerun
or replace the old 150-round reference training. A mismatch is fatal,
and an existing YAML is never overwritten.
The uncentered-basis reconstruction checks that its **centered** version
matches the old basis (minimum principal cosine .995), then alters
**only** centering. If the old checkpoints or parity are missing, the
preparation stops. No target data are used for basis fitting.

## Start on both servers

```bash
cd /data/deli/MFL-new/MFL-new
git fetch origin
git switch aslwf-final16-two-servers-20261022
conda activate mfl
```

**Server A (uncentered, eight FL runs; foreground live status):**

```bash
bash start_serverA_live.sh
```

**Server B (c=.9 frozen, eight FL runs; foreground live status):**

```bash
bash start_serverB_live.sh
```

Do **not** use `nohup`, redirect the commands to files, or add `&` if
you want to see the progress in your current terminal. The two local GPU
queues are launched automatically by each script and stream their outputs
concurrently to one screen with `[GPU 0]` / `[GPU 1]` prefixes.
Each completed communication round is reported as
`[FL PROGRESS] round=.../150 participants=12 elapsed_s=...`,
and per-GPU queue completion appears as `[QUEUE PROGRESS] ... 1/4 jobs`.
Validation AUROC appears when validation runs. This is a **real per-round
completion log**, not an estimate of how long a round will take.

Each GPU queue waits until `nvidia-smi` reports **at least 7,500 MiB
of free GPU memory** before launching a task. This is a startup
threshold, **not a guarantee** that a CLIP ViT-L/14 training task can
complete within 7.5 GiB; an out-of-memory error can still occur.
You may override the startup threshold, e.g.
`MIN_FREE_MIB=12000 bash start_serverA_live.sh`, but the default for
these two final16 launchers is 7,500 MiB. The general-purpose
`run_defense70_job.sh` default for other experiments is unchanged.

The detailed GPU queue logs and per-job attempt logs are **still saved**
at `logs/acm_revision/final16_20261022/{uncentered,c09}/`; the final
CP/CPT attack and original-model-space distortion audits also stream
their progress. If model initialization, basis fitting or GPU-memory
checks take time, the terminal shows the current preparatory phase.
For A, basis fitting must complete before its eight formal FL runs start.

To specify other physical GPU IDs, use e.g.
`GPU0=1 GPU1=3 bash start_serverB_live.sh`.
Launch **one** script per server. A per-phase `flock` guard now refuses
a duplicate launcher; the same GPU-memory checks, retry/resume checkpoint
mechanism, and completed-run skip behavior are retained from
`run_defense70_job.sh`. If you already launched a prior background copy,
check its processes before starting a second instance.


## CUDA driver preflight (mandatory)

The final16 launchers now call
`scripts/acm_revision/verify_final16_gpu.sh` **before** the first
scientific preparation or training job. Each selected physical GPU must
pass `torch.cuda.is_available()`, a CUDA tensor allocation, and device
synchronization in its own `CUDA_VISIBLE_DEVICES` context. If the
NVIDIA driver is too old for this PyTorch build, the launcher stops
rather than silently switching to CPU.

Diagnosis on either server:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Do **not** merely lower `MIN_FREE_MIB` to bypass an incompatible CUDA driver:
VRAM and CUDA compatibility are independent requirements. Existing
checkpoint files are preserved and never automatically deleted. If any
formal runs progressed on CPU due to the old automatic fallback, isolate
those CPU-generated results/checkpoints before resuming GPU training;
do not combine CPU and GPU trajectories in one formal seed result.
To fix incompatibility, ask the machine administrator to upgrade the
NVIDIA driver or use a pinned PyTorch build compatible with the
machine's current driver, consistently across both servers.

## Outputs and audits

Output directories are **separate** from the old supplemental branches:

```text
configs/acm_revision/generated/final16_20261022/{uncentered,c09}/
results/acm_revision/final16_20261022/{uncentered,c09}/
results/acm_revision/final16_20261022/audits/
logs/acm_revision/final16_20261022/{uncentered,c09}/
checkpoints/acm_revision/final16_20261022/{uncentered,c09}/
```

Each launcher verifies all 8 completed 150-round runs and runs a
nine-surface × CP/CPT × LR/RF/MLP defense-aware attack audit. It reports
five independent target seed AUROCs with mean, SD, bootstrap 95% CI.
The task evaluation chooses checkpoints by validation, not target metrics.

Server A also performs a **shadow-only matched raw-update L2 audit** of
the old PCA/random/matched/Complete-ASLWF baselines and the new uncentered
variant. It measures exact original-model-space distortion on identical
raw updates; these estimates must be identified as checkpoint-matched
diagnostics, not as actual historical target-run distortions.

For method-vs-no-defense task delta use a separately **verified** matching
no-defense CSV. Neither script invents baseline scores or silently treats
missing baselines as available.

The full technical guide is
`docs/paper_repro/supplement16_guide.md`; the Python entrypoints remain
`src.acm_revision.supplement16` and
`src.acm_revision.supplement16_distortion` for compatibility.

**Verification scope:** GitHub CI only tests code syntax and the static
16-job schedule. It does not run private dataset or GPU jobs. Do not
describe the new numerical results as obtained until all runs and audits
actually complete.

## Lightweight result export for analysis (no model weights)

After a run, or to inspect interim progress, update the source branch then
run on **each respective server**:

```bash
# Server A / Rheal
bash scripts/acm_revision/package_final16_analysis.sh A

# Server B / Eureka1
bash scripts/acm_revision/package_final16_analysis.sh B
```

The packages are saved separately as:

```text
exports/aslwf_final16_serverA_YYYYMMDD_HHMMSS_analysis.tar.gz
exports/aslwf_final16_serverB_YYYYMMDD_HHMMSS_analysis.tar.gz
```

Each includes the per-seed `summary.json`, `round_metrics.csv`,
`update_metadata.csv` (actual head/full L2), `capture_manifest.json`,
and `resolved_config.json`; generated YAML plans, frozen-manifest YAMLs,
per-target statistics and nine-surface CP/CPT attack CSV/JSON; and Server A's
matched-update distortion statistics when available.

No `.pt`, `.npz`, `.npy`, model weights, raw client updates, large logs,
private datasets or client sample-assignment manifest are included. Each
file has a default **25 MiB** limit. If a needed `update_metadata.csv`
exceeds this, rerun with `MAX_ANALYSIS_FILE_MIB=100` and inspect
`META/SKIPPED_FILES.txt`. The archive is a read-only copy, with no
effects on the original training processes or checkpoints.

`META/RUN_STATUS.csv` marks the eight expected per-server runs as
`PASS`, `MISSING`, `AMBIGUOUS` or `PARTIAL/NOT_VERIFIED`.
Only verified complete runs and complete audits should be used to draw
final numerical conclusions. Both servers' archives are needed for the
centered/uncentered and c=.9 publication-ready comparative analysis.
