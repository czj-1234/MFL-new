import argparse
import json
from pathlib import Path

from src.defense.subspace import CounterfactualSubspace
from src.defense.subspace import principal_angles_degrees
from src.defense.subspace import subspace_similarity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis-a", required=True)
    parser.add_argument("--basis-b", required=True)
    parser.add_argument("--basis-c", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    a = CounterfactualSubspace.load(args.basis_a).basis
    b = CounterfactualSubspace.load(args.basis_b).basis
    c = CounterfactualSubspace.load(args.basis_c).basis
    pairs = [("A-B", a, b), ("A-C", a, c), ("B-C", b, c)]
    results = []
    for name, left, right in pairs:
        results.append({
            "pair": name,
            "similarity": subspace_similarity(left, right),
            "angles": principal_angles_degrees(left, right).tolist(),
        })
    report = {"comparisons": results}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
