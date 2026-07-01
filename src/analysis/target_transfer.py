import argparse
import json
from pathlib import Path

import numpy as np

from src.defense.subspace import CounterfactualSubspace, removed_energy_ratio


def load_matrix(path):
    data = np.load(path)
    return data["updates"] if "updates" in data.files else data[data.files[0]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target-updates", required=True)
    p.add_argument("--basis", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    x = load_matrix(args.target_updates)
    u = CounterfactualSubspace.load(args.basis).basis
    ratios = removed_energy_ratio(x, u, alpha=1.0)
    report = {
        "num_updates": int(x.shape[0]),
        "update_dim": int(x.shape[1]),
        "mean_projection_energy": float(ratios.mean()),
        "std_projection_energy": float(ratios.std()),
        "median_projection_energy": float(np.median(ratios)),
        "min_projection_energy": float(ratios.min()),
        "max_projection_energy": float(ratios.max()),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
