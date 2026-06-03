# ============================================================
# Run 12 MVSA structural baseline experiments
# 3 settings × 4 association levels
# ============================================================

conda activate mfl

$settings = @(
    "image_only",
    "text_only",
    "modality_exclusive"
)

$associations = @(
    "iid",
    "0.3",
    "0.7",
    "1.0"
)

foreach ($setting in $settings) {
    foreach ($association in $associations) {

        Write-Host ""
        Write-Host "============================================================"
        Write-Host "Running setting=$setting, association=$association"
        Write-Host "============================================================"
        Write-Host ""

        python -m src.main `
            --setting $setting `
            --association $association `
            --rounds 200 `
            --samples_per_client 100

        if ($LASTEXITCODE -ne 0) {
            Write-Host ""
            Write-Host "Experiment failed: setting=$setting, association=$association"
            exit $LASTEXITCODE
        }
    }
}

Write-Host ""
Write-Host "All 12 experiments completed."