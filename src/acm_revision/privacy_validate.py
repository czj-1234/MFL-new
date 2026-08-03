from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ALL_GROUPS = [
    "classifier_bias",
    "classifier_weight",
    "classifier_head",
    "fusion",
    "image_encoder",
    "text_encoder",
    "missing_modality",
    "all_shared",
    "full_update",
]
SMOKE_GROUPS = ["classifier_head", "fusion", "full_update"]


def _successful_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if "error" not in frame.columns:
        return frame
    return frame[frame["error"].isna() | (frame["error"].astype(str).str.strip() == "")]


def validate_output(output_dir: str | Path, smoke: bool = False) -> dict:
    output_dir = Path(output_dir)
    groups = SMOKE_GROUPS if smoke else ALL_GROUPS
    failures = []
    details = {}

    canonical_path = output_dir / "canonical_leakage_summary.csv"
    label_path = output_dir / "label_distribution_summary.csv"
    structural_path = output_dir / "structural_privacy.csv"
    trajectory_path = output_dir / "trajectory_attacks.csv"
    report_path = output_dir / "postprocess_report.json"

    for path in [canonical_path, label_path, structural_path, trajectory_path, report_path]:
        if not path.exists() or path.stat().st_size == 0:
            failures.append(f"missing_or_empty:{path.name}")

    if failures:
        result = {"status": "FAIL", "output_dir": str(output_dir), "failures": failures}
        (output_dir / "validation_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        raise RuntimeError("; ".join(failures))

    canonical = pd.read_csv(canonical_path)
    label = pd.read_csv(label_path)
    structural = pd.read_csv(structural_path)
    trajectory = pd.read_csv(trajectory_path)
    post_report = json.loads(report_path.read_text(encoding="utf-8"))
    if post_report.get("status") != "PASS":
        failures.append("postprocess_report_not_pass")

    required_structural_columns = {
        "group",
        "update_norm_mean",
        "effective_rank",
        "within_label_cosine_mean",
        "between_label_cosine_mean",
        "cosine_separation_gap",
        "silhouette_score",
        "davies_bouldin_index",
        "calinski_harabasz_index",
    }
    missing_structural_columns = sorted(required_structural_columns - set(structural.columns))
    if missing_structural_columns:
        failures.append(f"structural_columns_missing:{missing_structural_columns}")

    for group in groups:
        group_detail = {}
        attack_path = output_dir / f"attack_summary__{group}.csv"
        leakage_path = output_dir / f"leakage_probabilities__{group}.csv"
        dist_pred_path = output_dir / f"label_distribution_predictions__{group}.csv"
        for path in [attack_path, leakage_path, dist_pred_path]:
            if not path.exists() or path.stat().st_size == 0:
                failures.append(f"{group}:missing_or_empty:{path.name}")

        canonical_rows = _successful_rows(canonical[canonical["group"] == group])
        label_rows = _successful_rows(label[label["group"] == group])
        structural_rows = _successful_rows(structural[structural["group"] == group])
        if canonical_rows.empty:
            failures.append(f"{group}:no_successful_canonical_leakage")
        if label_rows.empty:
            failures.append(f"{group}:no_successful_label_distribution")
        if structural_rows.empty:
            failures.append(f"{group}:no_successful_structural_rows")

        if attack_path.exists() and attack_path.stat().st_size > 0:
            attack = pd.read_csv(attack_path)
            successful = _successful_rows(attack)
            strict = successful
            for column, value in [
                ("protocol", "cross_partition"),
                ("target", "dominant_label"),
                ("attack_model", "logistic_regression"),
            ]:
                if column in strict.columns:
                    strict = strict[strict[column] == value]
                else:
                    strict = strict.iloc[0:0]
            if strict.empty:
                failures.append(f"{group}:strict_logistic_attack_missing_or_failed")
            group_detail["successful_attack_rows"] = int(len(successful))
            group_detail["strict_logistic_rows"] = int(len(strict))

        group_detail["canonical_rows"] = int(len(canonical_rows))
        group_detail["label_distribution_rows"] = int(len(label_rows))
        group_detail["structural_rows"] = int(len(structural_rows))
        details[group] = group_detail

    successful_trajectory = _successful_rows(trajectory)
    if successful_trajectory.empty:
        failures.append("no_successful_trajectory_attack")

    status = "PASS" if not failures else "FAIL"
    result = {
        "status": status,
        "output_dir": str(output_dir),
        "smoke": bool(smoke),
        "groups": groups,
        "successful_trajectory_rows": int(len(successful_trajectory)),
        "details": details,
        "failures": failures,
    }
    (output_dir / "validation_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    marker = output_dir / "VALIDATION_PASS"
    if status == "PASS":
        marker.write_text("PASS\n", encoding="utf-8")
    else:
        marker.unlink(missing_ok=True)
        raise RuntimeError("; ".join(failures))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate complete privacy post-processing outputs")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(json.dumps(validate_output(args.output_dir, smoke=args.smoke), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
