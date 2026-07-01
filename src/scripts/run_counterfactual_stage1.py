"""Run the first seed42 association/counterfactual pair and save update matrices.

This is the initial proof-of-concept run. It keeps the model/training seed fixed at
42 and compares association=0.7 with the balanced association=0.5 setting using
identical model and optimizer configuration.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.config import ExperimentArgs
from src.runner import run_one_experiment
from src.utils import load_config, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/config_hateful_counterfactual.yaml",
    )
    parser.add_argument(
        "--output-root",
        default="results/defense_seed42_assoc07/shadow/split_A",
    )
    return parser.parse_args()


def run_condition(cfg: dict, association: str, output_root: str) -> Path:
    # Reset every RNG before each paired run so model initialization and
    # stochastic training begin from the same seed.
    set_seed(42)

    args = ExperimentArgs(
        cfg,
        setting_name="modality_exclusive",
        association=association,
        output_root=output_root,
    )
    args.target_patterns = ["classifier"]

    _, _, all_mat = run_one_experiment(args)
    if all_mat is None:
        raise RuntimeError(
            f"No update matrix returned for association={association}. "
            "Check compute_structure_metrics()."
        )

    condition_dir = Path(args.out_dir)
    condition_dir.mkdir(parents=True, exist_ok=True)
    output_path = condition_dir / "classifier_updates.npz"
    np.savez_compressed(output_path, updates=np.asarray(all_mat))
    print(f"Saved classifier update matrix to: {output_path}")
    return output_path


def main() -> None:
    cli = parse_args()
    cfg = load_config(cli.config)

    associated_path = run_condition(
        cfg=cfg,
        association="0.7",
        output_root=cli.output_root,
    )
    counterfactual_path = run_condition(
        cfg=cfg,
        association="0.5",
        output_root=cli.output_root,
    )

    print("\nStage-1 paired runs complete.")
    print("Associated:    ", associated_path)
    print("Counterfactual:", counterfactual_path)
    print("\nLearn the rank-3 subspace with:")
    print(
        "python -m src.analysis.learn_counterfactual_subspace "
        f"--associated {associated_path} "
        f"--counterfactual {counterfactual_path} "
        "--rank 3 "
        "--output results/defense_seed42_assoc07/subspace/split_A_r3.npz"
    )


if __name__ == "__main__":
    main()
