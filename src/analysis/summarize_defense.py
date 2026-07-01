import argparse
import json
from pathlib import Path

import pandas as pd


FIELDS = [
    "setting_name", "association", "best_round",
    "best_test_acc", "best_test_macro_f1", "best_test_auroc",
    "test_acc", "test_macro_f1", "test_auroc",
    "e_rank", "Top1_ratio", "Top3_ratio", "Top5_ratio",
    "Silhouette", "DBI", "CHI", "kmeans_acc",
    "attack_success_rate_rf", "attack_success_rate_mlp",
    "attack_success_rate_xgb", "attack_success_rate_mean",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/defense_seed42_assoc07")
    parser.add_argument("--output", default="results/defense_seed42_assoc07/defense_summary.csv")
    args = parser.parse_args()

    root = Path(args.root)
    rows = []
    for path in root.rglob("summary.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        row = {"run": str(path.parent.relative_to(root))}
        for field in FIELDS:
            row[field] = data.get(field)
        rows.append(row)

    frame = pd.DataFrame(rows).sort_values("run") if rows else pd.DataFrame(columns=["run"] + FIELDS)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(frame.to_string(index=False))
    print("Saved:", output)


if __name__ == "__main__":
    main()
