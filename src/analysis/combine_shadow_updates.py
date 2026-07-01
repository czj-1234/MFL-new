import argparse
from pathlib import Path

import numpy as np


def load(path):
    data = np.load(path)
    return data["updates"] if "updates" in data.files else data[data.files[0]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--associated", nargs="+", required=True)
    p.add_argument("--counterfactual", nargs="+", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    if len(args.associated) != len(args.counterfactual):
        raise ValueError("Associated and counterfactual file counts must match")
    assoc = np.concatenate([load(x) for x in args.associated], axis=0)
    cf = np.concatenate([load(x) for x in args.counterfactual], axis=0)
    if assoc.shape != cf.shape:
        raise ValueError("Combined paired matrices must have the same shape")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "associated_all.npz", updates=assoc)
    np.savez_compressed(out / "counterfactual_all.npz", updates=cf)
    print("Combined shape:", assoc.shape)


if __name__ == "__main__":
    main()
