from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .advanced_attacks import run_informed_server_attack, run_multi_round_majority_vote
from .analysis import aggregate_independent_runs, bootstrap_ci, paired_regime_test
from .attack_tuning import tune_and_test_attack
from .attacks import make_attack_split, read_metadata
from .data_audit import audit_dataset_files
from .defense_sweep import generate_contrast_defense_sweep, generate_layerwise_adaptive_sweep
from .provenance import save_environment
from .secure_aggregation import build_secure_aggregate_observations


def _csv(value: str):
    return [x.strip() for x in value.split(",") if x.strip()]


def build_parser():
    parser = argparse.ArgumentParser(description="Advanced analysis tools for ACM revision experiments")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("tune-attack")
    p.add_argument("--metadata", nargs="+", required=True)
    p.add_argument("--protocol", default="cross_partition_temporal")
    p.add_argument("--group", default="classifier_head")
    p.add_argument("--model", required=True, choices=["logistic_regression", "linear_svm", "random_forest", "boosted_trees", "xgboost", "mlp"])
    p.add_argument("--target", default="dominant_label")
    p.add_argument("--observation", choices=["raw", "observed"], default="observed")
    p.add_argument("--selection-metric", choices=["macro_f1", "accuracy", "auroc"], default="macro_f1")
    p.add_argument("--pca-dim", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)

    p = sub.add_parser("informed-attack")
    p.add_argument("--metadata", nargs="+", required=True)
    p.add_argument("--protocol", default="cross_partition_temporal")
    p.add_argument("--group", default="full_update")
    p.add_argument("--model", default="mlp")
    p.add_argument("--target", default="dominant_label")
    p.add_argument("--observation", choices=["raw", "observed"], default="observed")
    p.add_argument("--include-client-identity", action="store_true")
    p.add_argument("--no-checkpoint", action="store_true")
    p.add_argument("--pca-dim", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)

    p = sub.add_parser("trajectory-vote")
    p.add_argument("--metadata", nargs="+", required=True)
    p.add_argument("--protocol", default="cross_partition")
    p.add_argument("--group", default="classifier_head")
    p.add_argument("--base-model", default="logistic_regression")
    p.add_argument("--target", default="dominant_label")
    p.add_argument("--observation", choices=["raw", "observed"], default="observed")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)

    p = sub.add_parser("audit-data")
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--output", required=True)

    p = sub.add_parser("secure-aggregation")
    p.add_argument("--metadata", nargs="+", required=True)
    p.add_argument("--group", default="full_update")
    p.add_argument("--observation", choices=["raw", "observed"], default="observed")
    p.add_argument("--output-dir", required=True)

    p = sub.add_parser("defense-sweep")
    p.add_argument("--base-config", required=True)
    p.add_argument("--basis", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--ranks", default="1,2,4,8,16")
    p.add_argument("--alphas", default="0,0.25,0.5,0.75,1.0")
    p.add_argument("--modes", default="classifier_head,all_shared,full_update")

    p = sub.add_parser("layerwise-sweep")
    p.add_argument("--base-config", required=True)
    p.add_argument("--basis", required=True)
    p.add_argument("--output-dir", required=True)

    p = sub.add_parser("bootstrap")
    p.add_argument("--csv", required=True)
    p.add_argument("--value", required=True)
    p.add_argument("--unit", required=True)
    p.add_argument("--n-bootstrap", type=int, default=5000)
    p.add_argument("--output", required=True)

    p = sub.add_parser("paired-test")
    p.add_argument("--csv", required=True)
    p.add_argument("--value", required=True)
    p.add_argument("--regime-column", default="setting_name")
    p.add_argument("--left", required=True)
    p.add_argument("--right", required=True)
    p.add_argument("--pair-columns", default="seed,concentration,num_clients,participation_rate")
    p.add_argument("--output", required=True)

    p = sub.add_parser("aggregate-runs")
    p.add_argument("--csv", required=True)
    p.add_argument("--metrics", required=True)
    p.add_argument("--groups", required=True)
    p.add_argument("--output", required=True)

    p = sub.add_parser("environment")
    p.add_argument("--output", required=True)

    return parser


def _write_json(payload: dict, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    args = build_parser().parse_args()

    if args.command in ("tune-attack", "informed-attack", "trajectory-vote"):
        frame = read_metadata(args.metadata)
        split = make_attack_split(frame, args.protocol, seed=args.seed)
        if args.command == "tune-attack":
            result = tune_and_test_attack(
                split,
                args.group,
                args.model,
                target=args.target,
                observation=args.observation,
                seed=args.seed,
                pca_dim=args.pca_dim,
                selection_metric=args.selection_metric,
                output_json=args.output,
            )
        elif args.command == "informed-attack":
            result = run_informed_server_attack(
                split,
                args.group,
                attack_model=args.model,
                target=args.target,
                observation=args.observation,
                seed=args.seed,
                pca_dim=args.pca_dim,
                include_client_identity=args.include_client_identity,
                include_checkpoint=not args.no_checkpoint,
            )
            _write_json(result, args.output)
        else:
            result = run_multi_round_majority_vote(
                split,
                args.group,
                base_attack_model=args.base_model,
                target=args.target,
                observation=args.observation,
                seed=args.seed,
            )
            _write_json(result, args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "audit-data":
        result = audit_dataset_files(args.train, args.val, args.test, args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "secure-aggregation":
        result = build_secure_aggregate_observations(args.metadata, args.group, args.observation, args.output_dir)
        print(result.to_string(index=False))
        return

    if args.command == "defense-sweep":
        ranks = [int(x) for x in _csv(args.ranks)]
        alphas = [float(x) for x in _csv(args.alphas)]
        result = generate_contrast_defense_sweep(
            args.base_config,
            args.basis,
            args.output_dir,
            ranks=ranks,
            alphas=alphas,
            modes=_csv(args.modes),
        )
        print(result.to_string(index=False))
        return

    if args.command == "layerwise-sweep":
        result = generate_layerwise_adaptive_sweep(args.base_config, args.basis, args.output_dir)
        print(result.to_string(index=False))
        return

    if args.command == "bootstrap":
        frame = pd.read_csv(args.csv)
        result = bootstrap_ci(frame, args.value, args.unit, n_bootstrap=args.n_bootstrap)
        _write_json(result, args.output)
        print(json.dumps(result, indent=2))
        return

    if args.command == "paired-test":
        frame = pd.read_csv(args.csv)
        result = paired_regime_test(
            frame,
            args.value,
            args.regime_column,
            args.left,
            args.right,
            pair_columns=_csv(args.pair_columns),
        )
        _write_json(result, args.output)
        print(json.dumps(result, indent=2))
        return

    if args.command == "aggregate-runs":
        frame = pd.read_csv(args.csv)
        result = aggregate_independent_runs(frame, _csv(args.metrics), _csv(args.groups))
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
        print(result.to_string(index=False))
        return

    if args.command == "environment":
        result = save_environment(args.output)
        print(json.dumps(result, indent=2))
        return


if __name__ == "__main__":
    main()
