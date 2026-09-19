from __future__ import annotations

import argparse

from .attacks import make_attack_split, read_metadata
from .auxiliary_sensitivity import run_auxiliary_data_sensitivity
from .baseline_sweep import generate_basis_baseline_sweep
from .clustering import heldout_cluster_diagnostics


def build_parser():
    parser = argparse.ArgumentParser(description="Supplemental ACM revision experiments")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("auxiliary-sensitivity")
    p.add_argument("--metadata", nargs="+", required=True)
    p.add_argument("--protocol", default="cross_partition_temporal")
    p.add_argument("--group", default="classifier_head")
    p.add_argument("--model", default="mlp")
    p.add_argument("--target", default="dominant_label")
    p.add_argument("--observation", choices=["raw", "observed"], default="observed")
    p.add_argument("--fractions", default="0.10,0.25,0.50,1.00")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)

    p = sub.add_parser("heldout-clustering")
    p.add_argument("--metadata", nargs="+", required=True)
    p.add_argument("--group", default="classifier_head")
    p.add_argument("--observation", choices=["raw", "observed"], default="observed")
    p.add_argument("--clusters", type=int, default=None)
    p.add_argument("--pca-dim", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)

    p = sub.add_parser("basis-baseline-sweep")
    p.add_argument("--base-config", required=True)
    p.add_argument("--pca", required=True)
    p.add_argument("--random", required=True)
    p.add_argument("--supervised", required=True)
    p.add_argument("--orthogonal", required=True)
    p.add_argument("--group", default="classifier_head")
    p.add_argument("--ranks", default="1,2,4,8,16")
    p.add_argument("--alphas", default="0.25,0.5,0.75,1.0")
    p.add_argument("--output-dir", required=True)
    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "auxiliary-sensitivity":
        frame = read_metadata(args.metadata)
        split = make_attack_split(frame, args.protocol, seed=args.seed)
        result = run_auxiliary_data_sensitivity(
            split,
            args.group,
            attack_model=args.model,
            target=args.target,
            observation=args.observation,
            fractions=[float(x) for x in args.fractions.split(",")],
            seed=args.seed,
            output_csv=args.output,
        )
        print(result.to_string(index=False))
        return

    if args.command == "heldout-clustering":
        result = heldout_cluster_diagnostics(
            args.metadata,
            args.group,
            observation=args.observation,
            n_clusters=args.clusters,
            pca_dim=args.pca_dim,
            seed=args.seed,
            output_csv=args.output,
        )
        print(result.to_string(index=False))
        return

    if args.command == "basis-baseline-sweep":
        result = generate_basis_baseline_sweep(
            args.base_config,
            basis_files={
                "rank_matched_pca": args.pca,
                "rank_matched_random_subspace": args.random,
                "supervised_sensitive_direction": args.supervised,
                "gradient_orthogonalization": args.orthogonal,
            },
            output_dir=args.output_dir,
            group=args.group,
            ranks=[int(x) for x in args.ranks.split(",")],
            alphas=[float(x) for x in args.alphas.split(",")],
        )
        print(result.to_string(index=False))
        return


if __name__ == "__main__":
    main()
