# ACM Transactions minimum revision experiment plan

## Decision rule

Run the strict cross-regime core matrix first. Retain the word **amplification** only if modality-exclusive training is significantly more vulnerable than image-only and text-only training under the same client/partition-disjoint protocols. Otherwise reposition the paper around dominant-label leakage, strict evaluation and matched diagnosis.

## Frozen strict data protocol

- Exact duplicate examples are removed.
- Near-duplicate image/text families remain indivisible and are assigned to one pool only.
- Shadow-train, shadow-validation and target samples are disjoint.
- Task validation/test examples are never used to construct clients.
- Raw examples are not reused across clients inside one FL run.
- Existing 40/20/40 pools are retained.

The core matrix uses 12 clients because the frozen shadow-validation pool contains only enough minority-label examples for a symmetric 12-client, 100-sample/client allocation. The real partitioner is executed for all 72 planned jobs during preflight; GPU work is blocked if any allocation is infeasible.

## Phase A: strict Core72 matrix

### Factors

- Settings: `image_only`, `text_only`, `modality_exclusive`
- Concentrations: `0.5`, `0.7`, `0.9`
- Communication rounds: 150
- Clients: 12
- Full participation
- Local epochs: 1
- Batch size: 16
- Aggregation: FedAvg
- Target FL seeds: 42, 43, 44, 45, 46
- Shadow-train FL seeds: 142, 143
- Shadow-validation FL seed: 242

### Per setting/concentration cell

- 2 independent shadow-train FL runs, 200 samples/client
- 1 independent shadow-validation FL run, 100 samples/client
- 5 independent target FL runs, 200 samples/client

Total: `3 settings × 3 concentrations × 8 runs = 72 FL runs`.

### Interpretation of concentration 0.5

The 0.5 condition retains nominal client roles only for attribution diagnostics. It is not interpreted as dominant-label privacy protection.

> The balanced condition is an attribution control rather than a privacy defense.

### Captured observations

Every round, exact:

- classifier bias
- classifier weight
- classifier head

At rounds 1, 5, 10, 20, 30, 50, 75, 100, 125 and 150:

- fusion and missing-modality parameters: exact
- image encoder, text encoder and all-shared groups: deterministic coordinate sketches
- full update: deterministic signed feature-hash projection in which every update coordinate contributes

The full-update projection is a compressed full-group observation, not an exact saved full vector.

For `modality_exclusive`, concentration 0.7, target seeds 42-44, reference checkpoints at rounds 1, 10, 30, 50, 75, 100, 125 and 150 are retained for identical-checkpoint matched contrasts.

### Core attack analysis

Pooled diagnostics:

- random-update
- temporal
- cross-client
- cross-partition
- cross-partition plus temporal

Attacks:

- random and majority
- update norm and cosine centroid
- logistic regression and linear SVM
- random forest, boosted trees and MLP
- temporal mean/variance, concatenation, LSTM, TCN and Transformer

Control targets:

- dominant label
- modality
- round group
- random balanced group
- persistent client identity under random-update/temporal diagnostics

Primary inferential results are computed separately for each of the five target FL seeds, with the target FL seed as the independent statistical unit. Bootstrap confidence intervals and paired modality-exclusive versus single-modality tests are produced after all nine cells are combined.

## Phase B: scale, participation and optimization robustness

Representative condition: Hateful Memes, modality-exclusive, concentration 0.7. Each configuration uses 1 shadow-train, 1 shadow-validation and 3 target runs.

### Client scale with approximately fixed total assigned data

- 20 clients × 60 samples
- 52 clients × 23 samples
- 100 clients × 12 samples

### Participation at 52 clients

- 10%
- 20%
- 50%
- 100% (reused from the client-scale block)

### One-factor robustness at 20 clients

- local epochs: 2 versus baseline 1
- batch size: 32 versus baseline 16
- optimizer: SGD versus AdamW
- aggregation: FedProx versus FedAvg
- smaller client data size: 30 versus 60 samples/client

Formal Phase B budget: 55 FL runs, with the full-participation 52-client baseline reused.

## Phase C: identical-checkpoint matched contrast

Use the retained Core72 reference checkpoints for modality-exclusive concentration 0.7, target seeds 42-44.

At each checkpoint, branch into concentrated and balanced local-update calculations while holding fixed:

- global checkpoint
- client identity and modality
- sample count
- optimizer state/initialization
- local steps
- batch order where feasible

Only label composition is changed. Report the result as a **label-composition-associated update contrast**, not a causal effect.

No additional reference FL run is required because the three reference trajectories are retained in Phase A. The branch calculations are local probes, not new global FL trajectories.

## Phase D: defense and privacy-utility curves

Representative condition: Hateful Memes, modality-exclusive, concentration 0.7, same 12-client protocol as Core72.

Compare:

- proposed contrast-aware filtering
- rank-matched PCA
- rank-matched random subspace
- clipping plus Gaussian noise
- no defense, reused from Core72

For each defense family evaluate two non-zero strengths; together with the shared no-defense point this yields at least a three-point privacy-utility curve. Every new operating point uses 1 shadow-train, 1 shadow-validation and 3 target runs. Add two target runs for the selected proposed point and strongest baseline so final comparisons use five target FL seeds.

Formal Phase D budget: 44 new FL runs.

All attack models are retrained on defended shadow updates. Rank, attenuation, clipping and noise are selected using shadow validation only.

## Phase E: additional dataset and architecture

- Dataset: MVSA-Multiple, original three-class task
- Architecture: ResNet18 + RoBERTa
- Settings: image-only, text-only, modality-exclusive
- Concentration: 0.7
- Per setting: 1 shadow-train, 1 shadow-validation, 3 target runs

Formal Phase E budget: 15 FL runs.

## Phase F: mechanism ablations

Representative Hateful Memes modality-exclusive concentration 0.7 condition. Each new configuration uses 1 shadow-train, 1 shadow-validation and 3 target runs.

- learned missing-modality embedding versus zero masking
- full fusion versus concatenation-only fusion
- fully trainable versus frozen encoders

Formal Phase F budget: 15 FL runs.

## Total minimum formal budget

- Phase A Core72: 72
- Phase B scale/participation/robustness: 55
- Phase C reference FL: 0 additional (reuses retained Phase A checkpoints)
- Phase D defense: 44
- Phase E dataset/architecture generalization: 15
- Phase F mechanism ablations: 15

Total minimum formal FL runs: **201**.

Only Phase A is launched initially. Phases B-F proceed after the Core72 decision point.
