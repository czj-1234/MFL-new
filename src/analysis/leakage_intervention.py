import argparse
import json
from pathlib import Path

import numpy as np

from src.defense.projection import pca_basis, random_basis
from src.defense.subspace import CounterfactualSubspace
from src.defense.subspace import project_onto_subspace, remove_subspace_component
from src.metrics import compute_structure_metrics


def load_matrix(path):
    data = np.load(path)
    return data["updates"] if "updates" in data.files else data[data.files[0]]


def records_from_matrix(matrix):
    records = []
    for index, row in enumerate(matrix):
        client_id = index % 4
        records.append({"update": row, "dominant_label": client_id % 2})
    return records


def evaluate(name, matrix, seed):
    metrics, _ = compute_structure_metrics(records_from_matrix(matrix), seed=seed)
    return {"name": name, **metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", required=True)
    parser.add_argument("--basis", required=True)
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    x = load_matrix(args.updates)
    proposed = CounterfactualSubspace.load(args.basis).basis[:, :args.rank]
    random_u = random_basis(x.shape[1], args.rank, args.seed)
    pca_u = pca_basis(x, args.rank)

    variants = {
        "baseline": x,
        "keep_sensitive": project_onto_subspace(x, proposed),
        "remove_sensitive": remove_subspace_component(x, proposed, args.alpha),
        "remove_random": remove_subspace_component(x, random_u, args.alpha),
        "remove_pca": remove_subspace_component(x, pca_u, args.alpha),
    }

    report = {"results": [evaluate(k, v, args.seed) for k, v in variants.items()]}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
