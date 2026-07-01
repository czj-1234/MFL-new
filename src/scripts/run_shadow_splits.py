import argparse
import json
from pathlib import Path

import numpy as np

from src.config import ExperimentArgs
from src.defense.runtime import predefined_clients
from src.defense.strict_counterfactual import build_strict_matched_pair
from src.runner import run_one_experiment
from src.utils import load_config, load_json, set_seed

SPLIT_SEEDS = {"A": 4201, "B": 4301, "C": 4401}


def run_one(cfg, split_name, tag, data, root):
    set_seed(42)
    args = ExperimentArgs(cfg, setting_name="modality_exclusive", association=tag,
                          output_root=str(Path(root) / ("split_" + split_name)))
    args.target_patterns = ["classifier"]
    with predefined_clients(data):
        _, _, matrix = run_one_experiment(args)
    if matrix is None:
        raise RuntimeError("No update matrix returned")
    path = Path(args.out_dir) / "classifier_updates.npz"
    np.savez_compressed(path, updates=np.asarray(matrix))
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config_hateful_counterfactual.yaml")
    p.add_argument("--splits", nargs="+", default=["A", "B", "C"])
    p.add_argument("--output-root", default="results/defense_seed42_assoc07/shadow")
    cli = p.parse_args()
    cfg = load_config(cli.config)
    train = load_json(cfg["data"]["train_json"])

    for split_name in cli.splits:
        assoc, cf, meta = build_strict_matched_pair(
            train, "modality_exclusive", 4, 2, 2000, 0.7, SPLIT_SEEDS[split_name]
        )
        split_dir = Path(cli.output_root) / ("split_" + split_name)
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "pair_metadata.json").write_text(json.dumps(meta, indent=2))
        a = run_one(cfg, split_name, "assoc_07", assoc, cli.output_root)
        c = run_one(cfg, split_name, "counterfactual", cf, cli.output_root)
        print(split_name, a, c)


if __name__ == "__main__":
    main()
