from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from .privacy_validate import ALL_GROUPS, SMOKE_GROUPS, validate_output


def _successful(frame: pd.DataFrame) -> pd.DataFrame:
    if "error" not in frame.columns:
        return frame
    return frame[frame["error"].isna() | (frame["error"].astype(str).str.strip() == "")]


def validate_core72_cell(
    output_dir: str | Path,
    target_seeds: Sequence[int] = (42, 43, 44, 45, 46),
    smoke: bool = False,
) -> dict:
    output_dir = Path(output_dir)
    failures: list[str] = []
    base = validate_output(output_dir, smoke=smoke)
    if base.get("status") != "PASS":
        failures.append("base_privacy_validation_failed")

    report_path = output_dir / "postprocess_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    matrix = report.get("matrix_validation") or {}
    if matrix.get("status") != "PASS":
        failures.append("run_matrix_validation_failed")

    task_path = output_dir / "task_metrics_per_target_run.csv"
    if not task_path.exists() or task_path.stat().st_size == 0:
        failures.append("task_metrics_per_target_run_missing")
        task = pd.DataFrame()
    else:
        task = pd.read_csv(task_path)
        observed = sorted(int(x) for x in task.get("target_seed", pd.Series(dtype=int)).unique())
        expected = sorted(int(x) for x in target_seeds)
        if observed != expected:
            failures.append(f"target_task_seeds_expected_{expected}_observed_{observed}")

    groups = SMOKE_GROUPS if smoke else ALL_GROUPS
    group_details = {}
    for group in groups:
        per_run_path = output_dir / f"strict_independent_runs__{group}.csv"
        summary_path = output_dir / f"strict_independent_summary__{group}.csv"
        if not per_run_path.exists() or per_run_path.stat().st_size == 0:
            failures.append(f"{group}:strict_independent_runs_missing")
            continue
        if not summary_path.exists() or summary_path.stat().st_size == 0:
            failures.append(f"{group}:strict_independent_summary_missing")
            continue

        frame = _successful(pd.read_csv(per_run_path))
        required_protocols = ["cross_partition"] if smoke else ["cross_partition", "cross_partition_temporal"]
        detail = {}
        for protocol in required_protocols:
            strict = frame[
                (frame.get("protocol") == protocol)
                & (frame.get("attack_model") == "logistic_regression")
                & (frame.get("target") == "dominant_label")
            ]
            observed = sorted(int(x) for x in strict.get("target_seed", pd.Series(dtype=int)).unique())
            expected = sorted(int(x) for x in target_seeds)
            if observed != expected:
                failures.append(
                    f"{group}:{protocol}:logistic_target_seeds_expected_{expected}_observed_{observed}"
                )
            detail[protocol] = {
                "successful_logistic_rows": int(len(strict)),
                "target_seeds": observed,
            }
        group_details[group] = detail

    balanced = bool(report.get("balanced_attribution_control"))
    concentration = float(report.get("concentration"))
    if concentration == 0.5 and not balanced:
        failures.append("c0.5_not_marked_as_balanced_attribution_control")
    if concentration != 0.5 and balanced:
        failures.append("non_balanced_condition_marked_as_balanced_control")

    status = "PASS" if not failures else "FAIL"
    result = {
        "status": status,
        "output_dir": str(output_dir),
        "smoke": bool(smoke),
        "target_seeds": [int(x) for x in target_seeds],
        "n_target_task_runs": int(len(task)),
        "groups": group_details,
        "failures": failures,
    }
    (output_dir / "core72_validation_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    marker = output_dir / "CORE72_VALIDATION_PASS"
    if status == "PASS":
        marker.write_text("PASS\n", encoding="utf-8")
    else:
        marker.unlink(missing_ok=True)
        raise RuntimeError("; ".join(failures))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate one complete Core72 post-processing cell")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-seeds", default="42,43,44,45,46")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    target_seeds = [int(x) for x in args.target_seeds.split(",") if x.strip()]
    print(
        json.dumps(
            validate_core72_cell(args.output_dir, target_seeds=target_seeds, smoke=args.smoke),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
