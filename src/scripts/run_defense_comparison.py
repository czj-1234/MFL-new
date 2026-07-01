import argparse
from pathlib import Path
import numpy as np

from src.config import ExperimentArgs
from src.defense.projection import pca_basis, random_basis
from src.defense.runtime import update_projection
from src.defense.subspace import CounterfactualSubspace
from src.runner import run_one_experiment
from src.utils import load_config, set_seed


def matrix(path):
    data = np.load(path)
    return data["updates"] if "updates" in data.files else data[data.files[0]]


def execute(cfg, method, basis, alpha, root):
    set_seed(42)
    args = ExperimentArgs(cfg, setting_name="modality_exclusive", association="0.7",
                          output_root=str(Path(root) / method))
    args.target_patterns = ["classifier"]
    if basis is None:
        _, _, updates = run_one_experiment(args)
    else:
        with update_projection(basis, alpha, ("classifier",)):
            _, _, updates = run_one_experiment(args)
    if updates is not None:
        np.savez_compressed(Path(args.out_dir) / "classifier_updates.npz", updates=np.asarray(updates))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config_hateful_counterfactual.yaml")
    p.add_argument("--method", default="all")
    p.add_argument("--basis", required=True)
    p.add_argument("--shadow-updates", required=True)
    p.add_argument("--rank", type=int, default=3)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--output-root", default="results/defense_seed42_assoc07/end_to_end")
    args = p.parse_args()

    cfg = load_config(args.config)
    shadow = matrix(args.shadow_updates)
    bases = {
        "baseline": None,
        "random": random_basis(shadow.shape[1], args.rank, 42),
        "pca": pca_basis(shadow, args.rank),
        "proposed": CounterfactualSubspace.load(args.basis).basis[:, :args.rank],
    }
    names = list(bases) if args.method == "all" else [args.method]
    for name in names:
        execute(cfg, name, bases[name], args.alpha, args.output_root)


if __name__ == "__main__":
    main()
