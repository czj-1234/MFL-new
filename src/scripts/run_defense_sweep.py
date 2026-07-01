import argparse
from pathlib import Path

import numpy as np

from src.config import ExperimentArgs
from src.defense.runtime import update_projection
from src.defense.subspace import CounterfactualSubspace
from src.runner import run_one_experiment
from src.utils import load_config, set_seed


def run(cfg, basis, rank, alpha, root):
    name = "proposed_r{}_a{}".format(rank, str(alpha).replace(".", ""))
    set_seed(42)
    args = ExperimentArgs(cfg, setting_name="modality_exclusive", association="0.7",
                          output_root=str(Path(root) / name))
    args.target_patterns = ["classifier"]
    with update_projection(basis[:, :rank], alpha, ("classifier",)):
        _, _, updates = run_one_experiment(args)
    if updates is not None:
        np.savez_compressed(Path(args.out_dir) / "classifier_updates.npz", updates=np.asarray(updates))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config_hateful_counterfactual.yaml")
    p.add_argument("--basis", required=True)
    p.add_argument("--output-root", default="results/defense_seed42_assoc07/sweep")
    p.add_argument("--alphas", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0])
    p.add_argument("--ranks", nargs="+", type=int, default=[1, 3, 5])
    args = p.parse_args()

    cfg = load_config(args.config)
    basis = CounterfactualSubspace.load(args.basis).basis
    done = set()
    for alpha in args.alphas:
        key = (3, alpha)
        if key not in done:
            run(cfg, basis, 3, alpha, args.output_root)
            done.add(key)
    best_alpha = 0.5
    for rank in args.ranks:
        key = (rank, best_alpha)
        if key not in done:
            run(cfg, basis, rank, best_alpha, args.output_root)
            done.add(key)


if __name__ == "__main__":
    main()
