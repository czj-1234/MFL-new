# ASLWF paper-aligned reproducibility package (2026-10-17)

## Status and scientific integrity

This branch is derived from acm-headfix-minimal (which already contains acm-transactions-revision). It is a **source-and-protocol reconstruction**, not a claim that the numerical tables have been independently rerun. It preserves the original code and completed experiment paths. Historical artifacts are not overwritten.

The paper of record is the 45-page manuscript "Introduction - 副本(20261017-032905).pdf" (2026-10-17). The manuscript PDF is not redistributed in this source branch. The code, manuscript and artifact provenance must be submitted together if the venue requires result-level verification.

**Do not assert verified reproduction until all PASS checks complete against the original frozen artifacts.** No numerical result should be pasted from the manuscript into generated result files. The audit is read-only with respect to original experiments.

## Branch ancestry

- Base: acm-headfix-minimal, commit 4b1b70186cba6a5ff2e4a24e5b8d8b638de9236b (2026-10-10).
- Predecessor: acm-transactions-revision, commit 9e4c9d42a53daa65f382901655eb00044a51adc5 (2026-09-19). The base is 18 commits ahead and contains that predecessor's history.
- Matched-contrast/Defense70: src/acm_revision/defense70_prepare.py, defense70_plan.py, contrast.py, hash_subspace.py, defense_dispatch.py.
- Attacker-guided completion: src/acm_revision/headfix_prepare.py, headfix_plan.py; scripts/acm_revision/prepare_headfix.sh, audit_headfix.py.
- Core leakage: src/acm_revision/core72_modality_first_plan.py, core72_postprocess.py and attack/geometry modules.

## Canonical experimental specification (manuscript Sections 3–5 and Appendices A–F)

| Item | Paper |
| --- | --- |
| Dataset | Hateful Memes, binary |
| Backbone | CLIP ViT-L/14, trainable image and text encoders, projected 512-d features, dropout 0.5 |
| Fusion | image, text, element-wise product, absolute difference; learned missing-modality embeddings |
| Regimes | image-only; text-only; modality-exclusive |
| Label concentration | 0.5, 0.7, 0.9; 0.5 is a *nominal-role* control, not a genuine dominant label |
| Modality-exclusive | 6 image and 6 text clients; per modality, 3 clients for each label role |
| Pool split | 40% shadow training / 20% shadow validation / 40% target after duplicate-family handling |
| Per-run clients | 12 persistent clients, full participation |
| Per-client data | 200 shadow-train / 100 shadow-val / 200 target; no within-run overlap |
| Federated rounds | 150; FedAvg; AdamW; 1 local epoch; lr 1e-6; wd 0.05; batch 16 |
| Seed roles | shadow train 142,143; shadow val 242; target 42–46 |
| Paper core | 3 regimes x 3 concentrations x (2 train + 1 val + 5 target) = 72 FL runs |
| Defense setting | modality-exclusive, c = 0.7 |
| Defense matched contrasts | 8 checkpoints x 12 client roles = 96 contrasts; 100 examples/branch |
| Matched reference checkpoints | 1, 10, 30, 50, 75, 100, 125, 150 |
| Paper attack milestones | 1, 10, 30, 50, 75, 100, 125, 150 |
| CPT training | shadow-train observations: 1, 10, 30, 50 |
| CPT selection | shadow-val observations: 75, 100, 125, 150 |
| CPT testing | target observations: 100, 125, 150 |
| CP | disjoint populations and trajectories, overlapping training stages permitted |
| Observation surfaces (9) | classifier_head, classifier_bias, classifier_weight, fusion, image_encoder, text_encoder, missing_modality, all_shared, full_update |
| Attack families | LR, RF, MLP; 9 x 3 x 2 (CP/CPT) = 54 candidate attacks |
| Reported attack metric | unflipped, shadow-validation-oriented mean of five separate target-run AUROCs; pooled target AUROC must be separately identified |
| Head / bias / weight recorded dims | 1026 / 6 / 3072 respectively; *not a nested tensor decomposition* |
| Fusion / missing | 1,311,744 / 1,024 exact coordinates |
| Image / text / all shared | 16,384 fixed deterministic coordinate-sketch coordinates |
| Full update paper attack | 2,048-d fixed signed feature-hash, every transmitted update coordinate contributes |
| Matched basis projected dimension | 512 signed-hash row-orthonormal projection (different from attack observation sketch) |
| Matched filter weak / strong | (rank,alpha)=(1,0.25) / (5,1.0) |
| Attacker-guided head | iterative raw-coordinate LR directions; candidate ranks tested on independent shadow validation; alpha=1 |
| Complete ASLWF | attacker-guided classifier-head basis **replaces** original head basis; fusion, image, text, missing groups retain strong matched-contrast bases |
| Independent target assessment | five completed defended FL trajectories, NOT post-hoc re-filtering of another method's saved updates |
| Utility | choose checkpoint by validation AUROC; report test AUROC at that checkpoint; no test selection |
| Uncertainty | independent FL runs (not client-round rows) are statistical units |

All transformations use the client's raw update before transmission; the server's aggregation remains FedAvg. No formal DP or unconditional privacy guarantee is claimed. Secure aggregation is not assumed.

## Canonical config generator (does not start training)

A new, non-destructive generator ensures that **prospective reproduction jobs** use the manuscript's exact eight analysis milestones and a 2,048-d full-update signed hash across all stages. It creates configs under configs/acm_revision/generated/paper_repro_20261017 and writes prospective results to a separate results/acm_revision/paper_repro_20261017 directory:

~~~bash
# 72 core jobs; reuses committed Core72 templates, needs no defense manifest
python -m src.acm_revision.paper_protocol_plan --stage core

# Requires real frozen results/acm_revision/defense70_prep/locked_params.yaml:
python -m src.acm_revision.paper_protocol_plan --stage baselines

# Requires BOTH real locked_params.yaml and locked_headfix.yaml:
python -m src.acm_revision.paper_protocol_plan --stage complete
~~~

Use --stage all only when both locked manifest files are available. The generator does not download data, train CLIP, fabricate basis vectors, or automatically reproduce manuscript metrics. It fails rather than substituting default ranks for selected ones. Generated job configs are new reproducibility trials; do not retroactively identify them with historical paper runs.

## Reproduction stages

1. **Data and core training.** Prepare Hateful Memes and strict de-duplicated disjoint pools using the existing strict_split / data_protocol utilities. Generate and run Core72 with the original launcher. Verify all 72 manifests, splits, configurations and captured updates.
2. **Contrast basis.** Use defense70_prepare against the original shadow-only reference. Check basis source (seed 142), 8 checkpoints, 12 client roles, same checkpoint per concentrated/balanced branch, sample budget 100, 96 contrasts, centered SVD, 512-dimensional high-dimensional projections and locked parameters. Check candidate-selection records on seed 242.
3. **Matched-only and controls.** Generate Defense70 jobs using the locked matched-basis manifest. The archived Defense70 matrix has 70 formal jobs for 10 operating points x 7 population/seed configurations; their available provenance must be supplied. Head-only here is *matched-contrast head*, not complete ASLWF head.
4. **Complete ASLWF head basis.** Prepare HeadFix from the original shadow Core72 updates: shadow train seeds 142 and 143, rounds <=60; independent shadow val seed 242, rounds >=61. The iterative LR-nullspace search in headfix_prepare.py refits after each removal. Save the selected rank, alpha, score and the immutable basis path in locked_headfix.yaml.
5. **Complete defended FL.** Generate a separate eight-run HeadFix matrix from both locked manifests. It uses two shadow-train, one shadow-val and five target runs. The classifier-head basis is replaced; other groups retain the strong matched basis. Every run must complete 150 rounds, have PASS capture, and remain isolated from No Defense and Matched-contrast Only training artifacts.
6. **Postprocess and attack auditing.** Strictly restrict *paper-protocol attacks* to the eight paper milestones. For CPT, test only late 100,125,150 target milestones. Retrain each LR/RF/MLP attack using the matching defended shadow updates. Fit scaling/PCA only on shadow training. Select surface, attack family, protocol and score orientation on shadow validation alone; evaluate selected configuration on five targets separately and report their **arithmetic mean**. Keep pooled target AUROC and target-side diagnostic max separately labeled.
7. **Verify metrics and table provenance.** Inspect the run-resolved utility and attack metrics, confidence intervals and exact capture representation definitions against the frozen manuscript. Use the traceability checklist and preflight script; unresolved discrepancies are blockers, not fields to fill by eye.

Example commands (from repository root; requires local original data and results):

~~~bash
python scripts/acm_revision/paper_repro_preflight.py --root .
python scripts/acm_revision/paper_repro_preflight.py --root . --strict-artifacts
python scripts/acm_revision/audit_headfix.py \
  --root results/acm_revision/headfix_r150 \
  --out results/acm_revision/paper_repro_headfix_audit \
  --paper-milestones \
  --baseline-task-csv results/acm_revision/core72_modality_first_v2_postprocess_r150/modality_exclusive__c0p7__r150/task_metrics_per_target_run.csv
~~~

The last command produces a **new paper-milestone audit**, not necessarily the same values as earlier all-round offline audits. Never overwrite, rename or relabel historical numerical outputs.

## Known paper-to-code differences requiring validation

1. **Attack observation calendar.** Historical HeadFix configs capture 10 milestone rounds: 1,5,10,20,30,50,75,100,125,150; the manuscript specifies 8. Extra *capture* rounds are acceptable only if the attack audit selects the paper's 8. Historical audit_headfix.py used all available observations before this branch added --paper-milestones. Without filtering, its CPT head attack can use every early and late round, contrary to the paper's eight-milestone attack protocol. Rerun and document the paper-aligned audit, do not assume published 0.6044 is reproduced.
2. **Full update representation.** Archived defense70_plan.py uses 8,192-d full-update hashing, whereas paper/HeadFix uses 2,048-d. The 8,192-d result is not a 2,048-d evaluation. Comparison under one observation protocol requires re-capture or documented separate postprocessing using original uncompressed transmitted updates. Do not assert identical raw inputs or back-convert hash sketches.
3. **Projection operators.** 16,384-d coordinate sketches and 2,048-d signed hash are attack observations; 512-d row-orthonormal signed hash is basis estimation/implicit projection. These are not interchangeable.
4. **Defended task provenance.** Same rounded task AUROC 0.658 for two methods does not imply shared training; verify five independent Complete ASLWF target summaries and individual task values. Publish selected locked_headfix.yaml and locked_params.yaml, not just default script constants.
5. **Core seed accounting.** Original two shadow-training seeds 142/143 plus one validation seed 242 plus five target seeds 42–46 produces 72 core conditions. Separate defense matrices are not counted as core72.
6. **Actual target metric.** Main paper uses per-target arithmetic mean, while an existing script can also print a pooled AUROC. The latter must never be silently treated as the former.
7. **Rank is not known from code defaults.** MAX_RANK=64 in headfix_prepare.py is a search upper bound. The selected rank is found only in the original locked_headfix.yaml; do not put 64 in manuscript as the final rank unless that manifest says so.
8. **Do not label signed hash as an exact/full uncompressed model vector.** It covers every original coordinate but compresses it to 2,048 dimensions.
9. **Scope.** The image pathway may retain recoverable label leakage even when classifier-head AUROC is reduced. The 54-config shadow-selected target metric is not a universal worst-case bound.
10. **Existing capture documentation.** docs/acm_revision/PRIVACY150_PROTOCOL.md is a historical/legacy protocol (including concentrations 0.6,0.8,1.0 and 10 milestones). It must NOT override the canonical paper specifications here.

## Submission handover: files that must accompany code

- Frozen raw/preprocessed dataset IDs or access instructions and strict pool/partition manifests (respect licenses, avoid uploading private raw data).
- Final commit SHA, environment export and hardware/software versions; core and defense generated configs.
- Original locked_params.yaml (matched basis), locked_headfix.yaml (actual chosen head rank), and basis NPZ or verifiable SHA256 references.
- Core72, Defense70 and HeadFix experiment summaries, capture manifests, per-seed test utility, per-target attack scores, and geometry.
- Table provenance file (see TRACEABILITY.csv) with per-table machine-readable artifacts; fixed-attack CPT/RF, surface-wise target maxima and shadow-selected 54-candidate results remain distinct.
- Run paper_repro_preflight.py with --strict-artifacts and archive its JSON PASS report; run paper-milestone attack audit and compare against paper values.

**Current status:** Source integration and protocol checker are provided; result-level equivalence is NOT certified until original frozen data and server results are available and the checks pass.
