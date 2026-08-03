# Core72 modality-first V2 execution

This execution namespace is intentionally isolated from all previous diagnostic, utility-only and earlier Core72 outputs.

## Ordering

All four GPU queues start with `modality_exclusive` jobs. Within each queue, every modality-exclusive job is completed before any `image_only` or `text_only` job.

Server A owns complete cells in this order:

1. `modality_exclusive @ 0.5`
2. `modality_exclusive @ 0.9`
3. `image_only @ 0.5`
4. `image_only @ 0.9`
5. `text_only @ 0.7`

Server B owns complete cells in this order:

1. `modality_exclusive @ 0.7`
2. `image_only @ 0.7`
3. `text_only @ 0.5`
4. `text_only @ 0.9`

Each cell remains on one physical server and contains 2 shadow-train, 1 shadow-validation and 5 target FL runs.

## Isolated paths

- Training results: `results/acm_revision/core72_modality_first_v2_r150/`
- Post-processing: `results/acm_revision/core72_modality_first_v2_postprocess_r150/`
- Final merged analysis: `results/acm_revision/core72_modality_first_v2_final/`
- Generated configs: `configs/acm_revision/generated/core72_modality_first_v2_r150/`
- Logs: `logs/acm_revision/core72_modality_first_v2_r150/`
- Smoke results: `results/acm_revision/core72_modality_first_v2_smoke_r2/`

No earlier result directory is used for completion checks, post-processing or aggregation.

## Commands

Server A:

```bash
bash scripts/acm_revision/run_core72_modality_first_serverA.sh 0 1
```

Server B:

```bash
bash scripts/acm_revision/run_core72_modality_first_serverB.sh 0 1
```

The legacy `run_core72_serverA.sh`, `run_core72_serverB.sh`, `run_core72_server.sh` and `smoke_test_core72.sh` entrypoints redirect to this V2 execution so an older command cannot accidentally write into the earlier Core72 namespace.
