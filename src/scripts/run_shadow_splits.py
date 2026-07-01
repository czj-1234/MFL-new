import argparse
import json
from pathlib import Path

import numpy as np

from src.config import ExperimentArgs
from src.defense.runtime import capture_raw_updates, predefined_clients
from src.defense.strict_counterfactual import build_strict_matched_pair
from src.runner import run_one_experiment
from src.utils import load_config, load_json, set_seed

SPLIT_SEEDS = {"A": 4201, "B": 4301, "C": 4401}


def run_one(cfg, split_name, tag, data, root):
    set_seed(42)

    args = ExperimentArgs(
        cfg,
        setting_name="modality_exclusive",
        association=tag,
        output_root=str(Path(root) / ("split_" + split_name)),
    )
    args.target_patterns = ["classifier"]

    captured = {}

    with predefined_clients(data), capture_raw_updates(captured):
        _, _, scaled_matrix = run_one_experiment(args)

    if scaled_matrix is None:
        raise RuntimeError(
            f"No scaled update matrix returned for split={split_name}, tag={tag}"
        )

    if "raw" not in captured:
        raise RuntimeError(
            f"Raw update matrix was not captured for split={split_name}, tag={tag}"
        )

    raw_matrix = np.asarray(captured["raw"], dtype=np.float64)
    scaled_matrix = np.asarray(scaled_matrix, dtype=np.float64)
    labels = np.asarray(captured["labels"], dtype=np.int64)

    if raw_matrix.shape != scaled_matrix.shape:
        raise ValueError(
            "Raw and scaled matrices have different shapes: "
            f"raw={raw_matrix.shape}, scaled={scaled_matrix.shape}"
        )

    if raw_matrix.shape[0] != labels.shape[0]:
        raise ValueError(
            "Update matrix and labels have different lengths: "
            f"updates={raw_matrix.shape[0]}, labels={labels.shape[0]}"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compatibility path used by existing downstream scripts.
    # It now contains RAW updates.
    default_path = out_dir / "classifier_updates.npz"
    raw_path = out_dir / "classifier_updates_raw.npz"
    scaled_path = out_dir / "classifier_updates_scaled.npz"
    labels_path = out_dir / "classifier_update_labels.npz"

    np.savez_compressed(
        default_path,
        updates=raw_matrix,
        labels=labels,
        representation="raw",
    )

    np.savez_compressed(
        raw_path,
        updates=raw_matrix,
        labels=labels,
        representation="raw",
    )

    np.savez_compressed(
        scaled_path,
        updates=scaled_matrix,
        labels=labels,
        representation="standard_scaled",
    )

    np.savez_compressed(
        labels_path,
        labels=labels,
    )

    print()
    print("=" * 72)
    print(f"Saved updates for split={split_name}, condition={tag}")
    print(f"Raw shape:    {raw_matrix.shape}")
    print(f"Scaled shape: {scaled_matrix.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Default raw:  {default_path}")
    print(f"Raw copy:     {raw_path}")
    print(f"Scaled:       {scaled_path}")
    print(f"Labels:       {labels_path}")
    print("=" * 72)
    print()

    return default_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/config_hateful_counterfactual.yaml",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["A", "B", "C"],
        default=["A", "B", "C"],
    )
    parser.add_argument(
        "--output-root",
        default="results/defense_seed42_assoc07/shadow",
    )
    cli = parser.parse_args()

    cfg = load_config(cli.config)
    train = load_json(cfg["data"]["train_json"])

    for split_name in cli.splits:
        print()
        print("#" * 80)
        print(f"Preparing strict matched split {split_name}")
        print("#" * 80)

        assoc, cf, meta = build_strict_matched_pair(
            train,
            "modality_exclusive",
            4,
            2,
            2000,
            0.7,
            SPLIT_SEEDS[split_name],
        )

        split_dir = Path(cli.output_root) / ("split_" + split_name)
        split_dir.mkdir(parents=True, exist_ok=True)

        (split_dir / "pair_metadata.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        associated_path = run_one(
            cfg,
            split_name,
            "assoc_07",
            assoc,
            cli.output_root,
        )

        counterfactual_path = run_one(
            cfg,
            split_name,
            "counterfactual",
            cf,
            cli.output_root,
        )

        print(f"Split {split_name} completed")
        print(f"Associated updates: {associated_path}")
        print(f"Counterfactual updates: {counterfactual_path}")


if __name__ == "__main__":
    main()
