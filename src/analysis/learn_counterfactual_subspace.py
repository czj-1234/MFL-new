from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.defense.subspace import (
    CounterfactualSubspace,
    fit_counterfactual_subspace,
    principal_angles_degrees,
    removed_energy_ratio,
    subspace_similarity,
)


def load_update_matrix(path: str) -> np.ndarray:
    """Load a 2D update matrix from .npy or .npz.

    For .npz files, the preferred key is ``updates``. If the archive contains
    only one array, that array is used automatically.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    if p.suffix == ".npy":
        matrix = np.load(p)
    elif p.suffix == ".npz":
        archive = np.load(p)
        if "updates" in archive.files:
            matrix = archive["updates"]
        elif len(archive.files) == 1:
            matrix = archive[archive.files[0]]
        else:
            raise ValueError(
                f"{p} contains multiple arrays {archive.files}; add an 'updates' key."
            )
    else:
        raise ValueError(f"Unsupported input format: {p.suffix}")

    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D update matrix, got {matrix.shape} from {p}.")
    return matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Learn a low-rank sensitive subspace from paired shadow updates."
    )
    parser.add_argument("--associated", required=True, help="Associated update matrix (.npy/.npz).")
    parser.add_argument("--counterfactual", required=True, help="Matched counterfactual matrix.")
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--output", required=True, help="Output .npz basis path.")
    parser.add_argument("--no-center", action="store_true")
    parser.add_argument(
        "--compare-basis",
        action="append",
        default=[],
        help="Optional learned basis .npz to compare against; may be repeated.",
    )
    parser.add_argument("--report", default=None, help="Optional JSON report path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    associated = load_update_matrix(args.associated)
    counterfactual = load_update_matrix(args.counterfactual)

    model = fit_counterfactual_subspace(
        associated_updates=associated,
        counterfactual_updates=counterfactual,
        rank=args.rank,
        center=not args.no_center,
    )
    model.save(args.output)

    differences = associated - counterfactual
    energy = removed_energy_ratio(differences, model.basis, alpha=1.0)

    report = {
        "associated_path": args.associated,
        "counterfactual_path": args.counterfactual,
        "num_pairs": int(associated.shape[0]),
        "update_dim": int(associated.shape[1]),
        "rank": int(model.rank),
        "singular_values": model.singular_values.tolist(),
        "mean_sensitive_energy_ratio": float(np.mean(energy)),
        "std_sensitive_energy_ratio": float(np.std(energy)),
        "comparisons": [],
    }

    for path in args.compare_basis:
        other = CounterfactualSubspace.load(path)
        report["comparisons"].append(
            {
                "basis_path": path,
                "similarity": subspace_similarity(model.basis, other.basis),
                "principal_angles_degrees": principal_angles_degrees(
                    model.basis, other.basis
                ).tolist(),
            }
        )

    report_path = Path(args.report) if args.report else Path(args.output).with_suffix(".json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved subspace: {args.output}")
    print(f"Saved report:   {report_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
