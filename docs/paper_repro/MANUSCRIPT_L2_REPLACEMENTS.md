# Manuscript L2 budget alignment — 2026-10-17 version

These are proposed textual edits for the uploaded 45-page PDF; the original PDF has not been modified. The code in this branch now *prospectively* enforces these rules. Historical run-by-run enforcement and whether all old target AUROCs remain unchanged cannot be established solely from the 12 shadow-validation mean L2 statistics. Do not present newly added code as retrospective proof of historical provenance.

## 1. Section 4.4 (page 19) — insert after the projection filtering definition, before "ASLWF applies a common low-rank filtering operation..."

For the evaluated subspace-filtering configurations, a per-client relative L2 perturbation budget is enforced after filtering and before transmission. Let \(\Delta_{i,t}^{\mathrm{raw}}\) and \(\Delta_{i,t}^{\mathrm{fil}}\) denote the original and candidate filtered updates. For \(g\in\{C,\mathrm{full}\}\), where \(P_C\) extracts the complete classifier head and \(P_{\mathrm{full}}\) retains all original update coordinates, define
\[
\rho_g=\frac{\|P_g(\Delta_{i,t}^{\mathrm{fil}}-\Delta_{i,t}^{\mathrm{raw}})\|_2}{\|P_g(\Delta_{i,t}^{\mathrm{raw}})\|_2+10^{-12}}.
\]
The same limits \(B_C=B_{\mathrm{full}}=1\) apply to the weak and strong filtering configurations. If either limit is exceeded, the filtering perturbation is uniformly rescaled by \(\eta=\min(1,\min_{g:\rho_g>0}B_g/\rho_g)\), and the transmitted update becomes \(\Delta_{i,t}^{\mathrm{raw}}+\eta(\Delta_{i,t}^{\mathrm{fil}}-\Delta_{i,t}^{\mathrm{raw}})\). Otherwise, the candidate filtered update is transmitted unchanged. This constraint is evaluated in original parameter coordinates, independently of the 2,048-dimensional signed-hash representation used for the full-update attack observation. FedAvg aggregation remains unchanged.

## 2. Section 4.4 (page 20) — replace the last baseline-comparison paragraph

Original starts: "These controls examine component coverage, subspace estimation, and update perturbation. The matched-contrast, PCA-based, and random-subspace baselines use identical ranks and attenuation strengths..."

Replacement:

These controls examine parameter-group coverage, subspace estimation, and update perturbation. The matched-contrast, PCA-based, and random-subspace baselines use identical ranks and attenuation strengths within each filtering regime and share the same per-client relative L2 perturbation limits. The imposed limits do not require identical realized update distortions across methods, since their selected directions and filtering scopes differ. Clip+Gaussian retains its separately calibrated clipping and perturbation parameters.

## 3. Appendix D.3 (page 41) — replace the paragraph immediately below Table D3

Original:

"The PCA and random controls use the same ranks and attenuation strengths as the corresponding matched-contrast configurations, with a common prescribed L2 perturbation budget within each regime. The realized update distortions are not assumed to be identical. Together with the global-basis and clipping-based controls, these baselines provide comparisons across subspace construction, filtering scope, and perturbation strategies."

Replacement:

The matched-contrast, PCA-based, and random-subspace filters use matched ranks and attenuation strengths within each regime. A common per-client relative L2 perturbation budget is additionally applied to the exact classifier-head update and the original full-model update, with an upper bound of 1 for each. The bounds are checked after filtering, and any excess filtering perturbation is uniformly rescaled before transmission. The 2,048-dimensional full-update signed-hash representation is used for attack observation and supplementary distortion diagnostics, not for enforcing the original-space budget. Consequently, methods may have different realized distortions even when they satisfy the same bounds. The Clip+Gaussian baseline retains its independent clipping and Gaussian-noise configuration.

## 4. Section 6 (page 30) — replace the rank/distortion limitation sentence

Original starts: "The matched-contrast, PCA-based, and random-subspace baselines use the same ranks and attenuation strengths..."

Replacement:

Although the matched-contrast, PCA-based, and random-subspace baselines use matched ranks and attenuation strengths under common per-client relative L2 perturbation limits, their realized distortions may still differ because they operate on different subspaces and parameter scopes.

## Interpretation and verification

- The L2 constraint is on **filter-induced change**, not the magnitude of the transmitted update itself.
- The classifier-head budget is measured on its exact coordinates. The full-update budget is measured on all original floating update coordinates, not the compressed 2048-dimensional attack sketch.
- The new runtime checks each local update and records the before/after ratios and whether rescaling happened.
- The historical per-method summary table reports averages from a single shadow-validation run, not per-update maxima; these summaries cannot alone certify historical enforcement or unchanged historical numerical results.
