from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import layer_geometry_report
from .attacks import (
    make_attack_split,
    read_metadata,
    run_attack_grid,
    run_label_distribution_inference,
    run_neural_trajectory_attack,
    run_trajectory_statistical_attack,
)
from .basis_baselines import (
    fit_gradient_orthogonalization_basis,
    fit_pca_basis,
    fit_random_basis,
    fit_supervised_sensitive_basis,
)
from .contrast import collect_identical_checkpoint_contrasts, fit_contrast_bases
from .data_protocol import build_strict_pools, load_json, save_strict_pools
from .fl import run_strict_fl
from .orchestrator import generate_job_files, load_yaml, select_defense_operating_point
from .privacy import rdp_epsilon


def _csv_list(value: str):
    return [x.strip() for x in value.split(",") if x.strip()]


def _int_list(value: str):
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ACM Transactions strict revision experiment suite")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare-data", help="Create disjoint shadow-train/shadow-val/target pools")
    p.add_argument("--config", required=True)

    p = sub.add_parser("generate-jobs", help="Expand an E1-E11 matrix profile into resolved YAML jobs")
    p.add_argument("--matrix", required=True)
    p.add_argument("--profile", required=True)
    p.add_argument("--output-dir", default="configs/acm_revision/generated")

    p = sub.add_parser("run-fl", help="Run one strict independent FL experiment")
    p.add_argument("--config", required=True)
    p.add_argument("--population", choices=["shadow_train", "shadow_val", "target"], default=None)

    p = sub.add_parser("run-attacks", help="Run tabular strict attacks over saved update metadata")
    p.add_argument("--metadata", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--protocols", default="random_update,temporal,cross_client,cross_partition,cross_partition_temporal")
    p.add_argument("--groups", default="classifier_bias,classifier_weight,classifier_head,fusion,image_encoder,text_encoder,all_shared,full_update")
    p.add_argument("--models", default="random,majority,update_norm,cosine_centroid,logistic_regression,linear_svm,random_forest,boosted_trees,xgboost,mlp")
    p.add_argument("--targets", default="dominant_label,modality,client_id,round,random_group")
    p.add_argument("--observation", choices=["raw", "observed"], default="observed")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pca-dim", type=int, default=256)

    p = sub.add_parser("label-distribution", help="Evaluate full label-distribution inference followed by argmax")
    p.add_argument("--metadata", nargs="+", required=True)
    p.add_argument("--protocol", default="cross_partition_temporal")
    p.add_argument("--group", default="classifier_head")
    p.add_argument("--num-classes", type=int, required=True)
    p.add_argument("--observation", choices=["raw", "observed"], default="observed")
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)

    p = sub.add_parser("trajectory", help="Run trajectory statistical/LSTM/TCN/Transformer attacks")
    p.add_argument("--metadata", nargs="+", required=True)
    p.add_argument("--protocol", default="cross_partition")
    p.add_argument("--group", default="classifier_head")
    p.add_argument("--methods", default="temporal_mean,temporal_variance,mean_variance,concatenation,lstm,tcn,transformer")
    p.add_argument("--target", default="dominant_label")
    p.add_argument("--observation", choices=["raw", "observed"], default="observed")
    p.add_argument("--sequence-length", type=int, default=10)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)

    p = sub.add_parser("collect-contrast", help="Collect identical-checkpoint concentrated-vs-balanced contrasts")
    p.add_argument("--config", required=True)
    p.add_argument("--reference-run-dir", required=True)
    p.add_argument("--pool-json", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--rounds", required=True, help="Comma-separated checkpoint rounds")
    p.add_argument("--groups", default="classifier_head,fusion,all_shared,full_update")
    p.add_argument("--concentrated", type=float, required=True)
    p.add_argument("--balanced", type=float, default=None)
    p.add_argument("--clients", default=None)
    p.add_argument("--samples-per-branch", type=int, default=None)

    p = sub.add_parser("fit-basis", help="Collect matched contrasts and fit an ordered SVD basis")
    p.add_argument("--config", required=True)
    p.add_argument("--reference-run-dir", required=True)
    p.add_argument("--pool-json", required=True)
    p.add_argument("--contrast-dir", required=True)
    p.add_argument("--rounds", required=True)
    p.add_argument("--groups", default="classifier_head,fusion,all_shared,full_update")
    p.add_argument("--concentrated", type=float, required=True)
    p.add_argument("--balanced", type=float, default=None)
    p.add_argument("--max-rank", type=int, default=None)
    p.add_argument("--output", required=True)

    p = sub.add_parser("fit-baseline-basis", help="Fit PCA/random/supervised/orthogonalization defense basis on shadow updates")
    p.add_argument("--method", required=True, choices=["pca", "random", "supervised", "orthogonal"])
    p.add_argument("--metadata", nargs="+", default=None)
    p.add_argument("--group", default="classifier_head")
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--dimension", type=int, default=None, help="Required only for random basis without metadata")
    p.add_argument("--observation", choices=["raw", "observed"], default="raw")
    p.add_argument("--target", default="dominant_label")
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)

    p = sub.add_parser("mechanism", help="Compute layer-wise norms/cosines/SVD/effective-rank diagnostics")
    p.add_argument("--metadata", nargs="+", required=True)
    p.add_argument("--groups", default="classifier_bias,classifier_weight,classifier_head,fusion,image_encoder,text_encoder,all_shared,full_update")
    p.add_argument("--observation", choices=["raw", "observed"], default="observed")
    p.add_argument("--output", required=True)

    p = sub.add_parser("select-defense", help="Select rank/alpha/defense point on shadow-validation only")
    p.add_argument("--validation-results", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--attack-metric", default="attack_asr")
    p.add_argument("--utility-metric", default="task_auroc")
    p.add_argument("--max-utility-drop", type=float, default=0.02)

    p = sub.add_parser("dp-epsilon", help="Report RDP epsilon for clipping + Gaussian-noise baseline")
    p.add_argument("--noise-multiplier", type=float, required=True)
    p.add_argument("--sample-rate", type=float, required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--delta", type=float, required=True)

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "prepare-data":
        cfg = load_yaml(args.config)
        data_cfg = cfg["data"]
        train = load_json(data_cfg["train_json"])
        val = load_json(data_cfg["val_json"])
        test = load_json(data_cfg["test_json"])
        split_cfg = cfg.get("strict_split", {})
        pools, report = build_strict_pools(
            train,
            val,
            test,
            shadow_train_ratio=float(split_cfg.get("shadow_train_ratio", 0.4)),
            shadow_val_ratio=float(split_cfg.get("shadow_val_ratio", 0.2)),
            target_ratio=float(split_cfg.get("target_ratio", 0.4)),
            seed=int(cfg["seed"]),
            group_key=split_cfg.get("group_key"),
            deduplicate=bool(split_cfg.get("deduplicate", True)),
            enable_near_duplicate_check=bool(split_cfg.get("near_duplicate_check", True)),
        )
        save_strict_pools(pools, report, data_cfg["pools_dir"])
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    if args.command == "generate-jobs":
        manifest = generate_job_files(args.matrix, args.profile, args.output_dir)
        print(manifest.to_string(index=False))
        return

    if args.command == "run-fl":
        cfg = load_yaml(args.config)
        population = args.population or cfg.get("experiment", {}).get("population", "target")
        print(json.dumps(run_strict_fl(cfg, population=population), ensure_ascii=False, indent=2))
        return

    if args.command == "run-attacks":
        result = run_attack_grid(
            args.metadata,
            args.output,
            protocols=_csv_list(args.protocols),
            groups=_csv_list(args.groups),
            models=_csv_list(args.models),
            targets=_csv_list(args.targets),
            observation=args.observation,
            seed=args.seed,
            pca_dim=args.pca_dim,
        )
        print(result.to_string(index=False))
        return

    if args.command == "label-distribution":
        frame = read_metadata(args.metadata)
        split = make_attack_split(frame, args.protocol, seed=args.seed)
        result = run_label_distribution_inference(split, args.group, args.num_classes, observation=args.observation, seed=args.seed)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([result]).to_csv(args.output, index=False)
        print(json.dumps(result, indent=2))
        return

    if args.command == "trajectory":
        frame = read_metadata(args.metadata)
        split = make_attack_split(frame, args.protocol, seed=args.seed)
        rows = []
        for method in _csv_list(args.methods):
            if method in ("lstm", "tcn", "transformer"):
                result = run_neural_trajectory_attack(
                    split,
                    args.group,
                    method,
                    target=args.target,
                    observation=args.observation,
                    sequence_length=args.sequence_length,
                    seed=args.seed,
                )
            else:
                result = run_trajectory_statistical_attack(
                    split,
                    args.group,
                    method,
                    target=args.target,
                    observation=args.observation,
                    sequence_length=args.sequence_length,
                    seed=args.seed,
                )
            rows.append(result)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.output, index=False)
        print(pd.DataFrame(rows).to_string(index=False))
        return

    if args.command in ("collect-contrast", "fit-basis"):
        cfg = load_yaml(args.config)
        output_dir = args.contrast_dir if args.command == "fit-basis" else args.output_dir
        result = collect_identical_checkpoint_contrasts(
            cfg,
            args.reference_run_dir,
            args.pool_json,
            output_dir,
            checkpoint_rounds=_int_list(args.rounds),
            groups=_csv_list(args.groups),
            concentrated=args.concentrated,
            balanced=args.balanced,
            clients=_int_list(args.clients) if getattr(args, "clients", None) else None,
            samples_per_branch=getattr(args, "samples_per_branch", None),
        )
        if args.command == "fit-basis":
            diagnostics = fit_contrast_bases(result, args.output, max_rank=args.max_rank)
            print(json.dumps(diagnostics, indent=2))
        else:
            print(f"Collected {len(result['records'])} matched contrast records.")
        return

    if args.command == "fit-baseline-basis":
        if args.method != "random" and not args.metadata:
            raise SystemExit("--metadata is required for pca/supervised/orthogonal basis fitting")
        if args.method == "pca":
            result = fit_pca_basis(args.metadata, args.group, args.output, args.rank, args.observation)
        elif args.method == "supervised":
            result = fit_supervised_sensitive_basis(
                args.metadata,
                args.group,
                args.output,
                args.rank,
                target=args.target,
                observation=args.observation,
                seed=args.seed,
            )
        elif args.method == "orthogonal":
            result = fit_gradient_orthogonalization_basis(args.metadata, args.group, args.output, args.rank, args.observation)
        else:
            dimension = args.dimension
            if dimension is None:
                if not args.metadata:
                    raise SystemExit("random basis needs --dimension or --metadata")
                frame = read_metadata(args.metadata)
                first = np.load(frame.iloc[0]["update_path"], allow_pickle=False)
                key = f"{args.observation}__{args.group}"
                dimension = int(np.asarray(first[key]).size)
                first.close()
            result = fit_random_basis(dimension, args.output, args.rank, seed=args.seed)
        print(json.dumps(result, indent=2))
        return

    if args.command == "mechanism":
        result = layer_geometry_report(args.metadata, _csv_list(args.groups), args.observation, args.output)
        print(result.to_string(index=False))
        return

    if args.command == "select-defense":
        result = select_defense_operating_point(
            args.validation_results,
            args.output,
            attack_metric=args.attack_metric,
            utility_metric=args.utility_metric,
            max_utility_drop=args.max_utility_drop,
        )
        print(json.dumps(result, indent=2))
        return

    if args.command == "dp-epsilon":
        result = {
            "epsilon": rdp_epsilon(args.noise_multiplier, args.sample_rate, args.steps, args.delta),
            "delta": args.delta,
            "noise_multiplier": args.noise_multiplier,
            "sample_rate": args.sample_rate,
            "steps": args.steps,
        }
        print(json.dumps(result, indent=2))
        return


if __name__ == "__main__":
    main()
