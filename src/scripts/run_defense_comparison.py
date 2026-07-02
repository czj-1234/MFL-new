import argparse
from pathlib import Path

import numpy as np

from src.config import ExperimentArgs
from src.defense.projection import pca_basis, random_basis
from src.defense.runtime import capture_raw_updates, update_projection
from src.defense.subspace import CounterfactualSubspace
from src.runner import run_one_experiment
from src.utils import load_config, set_seed


def matrix(path):
    data = np.load(path)
    return data["updates"] if "updates" in data.files else data[data.files[0]]


def value_tag(value):
    return str(value).replace(".", "p")


def execute(cfg, method, basis, alpha, root, gaussian_noise_ratio):
    set_seed(42)

    args = ExperimentArgs(
        cfg,
        setting_name="modality_exclusive",
        association="0.7",
        output_root=str(Path(root) / method),
    )
    args.target_patterns = ["classifier"]

    captured = {}
    if basis is None:
        if gaussian_noise_ratio > 0:
            raise ValueError(
                "Gaussian perturbation is applied after subspace projection; "
                "use --method proposed, random, or pca instead of baseline."
            )
        with capture_raw_updates(captured):
            _, _, scaled_updates = run_one_experiment(args)
    else:
        with update_projection(
            basis,
            alpha,
            ("classifier",),
            gaussian_noise_ratio=gaussian_noise_ratio,
            noise_seed=42,
        ), capture_raw_updates(captured):
            _, _, scaled_updates = run_one_experiment(args)

    if scaled_updates is None:
        raise RuntimeError(f"No scaled updates returned for method={method}")
    if "raw" not in captured:
        raise RuntimeError(f"No raw updates captured for method={method}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_updates = np.asarray(captured["raw"], dtype=np.float64)
    scaled_updates = np.asarray(scaled_updates, dtype=np.float64)
    labels = np.asarray(captured["labels"], dtype=np.int64)

    np.savez_compressed(
        out_dir / "classifier_updates.npz",
        updates=raw_updates,
        labels=labels,
        representation="raw",
        gaussian_noise_ratio=float(gaussian_noise_ratio),
        formal_dp_guarantee=False,
    )
    np.savez_compressed(
        out_dir / "classifier_updates_raw.npz",
        updates=raw_updates,
        labels=labels,
        representation="raw",
        gaussian_noise_ratio=float(gaussian_noise_ratio),
        formal_dp_guarantee=False,
    )
    np.savez_compressed(
        out_dir / "classifier_updates_scaled.npz",
        updates=scaled_updates,
        labels=labels,
        representation="standard_scaled",
        gaussian_noise_ratio=float(gaussian_noise_ratio),
        formal_dp_guarantee=False,
    )
    np.savez_compressed(
        out_dir / "classifier_update_labels.npz",
        labels=labels,
    )

    print(f"Completed method={method}")
    print(f"Gaussian noise ratio={gaussian_noise_ratio}")
    print("Formal DP guarantee=False")
    print(f"Output directory: {out_dir}")
    print(f"Raw shape: {raw_updates.shape}")
    print(f"Scaled shape: {scaled_updates.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/config_hateful_counterfactual.yaml",
    )
    parser.add_argument(
        "--method",
        choices=["all", "baseline", "random", "pca", "proposed"],
        default="all",
    )
    parser.add_argument("--basis", required=True)
    parser.add_argument("--shadow-updates", required=True)
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument(
        "--gaussian-noise-ratio",
        type=float,
        default=0.0,
        help=(
            "Gaussian noise standard deviation as a fraction of projected-update "
            "RMS. This is empirical perturbation, not formal differential privacy."
        ),
    )
    parser.add_argument(
        "--output-root",
        default="results/defense_seed42_assoc07/end_to_end",
    )
    args = parser.parse_args()

    if args.gaussian_noise_ratio < 0:
        raise ValueError("gaussian-noise-ratio must be non-negative")
    if args.method == "all" and args.gaussian_noise_ratio > 0:
        raise ValueError(
            "With Gaussian perturbation, run one projected method at a time "
            "(for example, --method proposed)."
        )

    cfg = load_config(args.config)
    shadow = matrix(args.shadow_updates)

    learned = CounterfactualSubspace.load(args.basis).basis
    if args.rank < 1 or args.rank > learned.shape[1]:
        raise ValueError(
            f"rank must be in [1, {learned.shape[1]}], got {args.rank}"
        )

    run_name = f"rank_{args.rank}_alpha_{value_tag(args.alpha)}"
    if args.gaussian_noise_ratio > 0:
        run_name += f"_noise_{value_tag(args.gaussian_noise_ratio)}"
    run_root = Path(args.output_root) / run_name

    bases = {
        "baseline": None,
        "random": random_basis(shadow.shape[1], args.rank, 42),
        "pca": pca_basis(shadow, args.rank),
        "proposed": learned[:, :args.rank],
    }

    names = list(bases) if args.method == "all" else [args.method]

    print("=" * 72)
    print("End-to-end defense comparison")
    print(f"rank={args.rank}, alpha={args.alpha}")
    print(f"gaussian_noise_ratio={args.gaussian_noise_ratio}")
    print("formal_dp_guarantee=False")
    print(f"output root={run_root}")
    print(f"methods={names}")
    print("=" * 72)

    for name in names:
        execute(
            cfg,
            name,
            bases[name],
            args.alpha,
            run_root,
            args.gaussian_noise_ratio,
        )


if __name__ == "__main__":
    main()
