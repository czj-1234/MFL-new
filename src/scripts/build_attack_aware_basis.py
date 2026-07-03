import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.defense.subspace import CounterfactualSubspace


def _load_updates_and_labels(path):
    data = np.load(path)
    updates = data["updates"] if "updates" in data.files else data[data.files[0]]
    if "labels" not in data.files:
        raise ValueError(
            f"{path} does not contain labels. Use a shadow-update file saved with labels."
        )
    labels = data["labels"]
    updates = np.asarray(updates, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if updates.ndim != 2:
        raise ValueError(f"Expected a 2D update matrix, got {updates.shape}")
    if labels.ndim != 1 or labels.shape[0] != updates.shape[0]:
        raise ValueError(
            f"Label shape {labels.shape} is incompatible with updates {updates.shape}"
        )
    return updates, labels


def _append_orthogonal(columns, vector, tolerance=1e-10):
    vector = np.asarray(vector, dtype=np.float64).reshape(-1)
    for column in columns:
        vector = vector - column * float(column @ vector)
    norm = float(np.linalg.norm(vector))
    if norm > tolerance:
        columns.append(vector / norm)


def build_attack_aware_basis(updates, labels, counterfactual_basis, rank, seed=42):
    classes = np.unique(labels)
    if classes.size != 2:
        raise ValueError(
            "The current attack-aware builder expects a binary leakage label. "
            f"Found classes={classes.tolist()}"
        )

    mean = updates.mean(axis=0)
    scale = updates.std(axis=0)
    scale[scale < 1e-12] = 1.0
    standardized = (updates - mean) / scale

    attack_model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=5000,
        random_state=seed,
        solver="liblinear",
    )
    attack_model.fit(standardized, labels)

    # Convert the linear attack direction from standardized coordinates back to
    # the raw classifier-update coordinate system.
    attack_direction = attack_model.coef_[0] / scale

    class_zero = updates[labels == classes[0]].mean(axis=0)
    class_one = updates[labels == classes[1]].mean(axis=0)
    mean_difference = class_one - class_zero

    columns = []

    # Put supervised leakage directions first, then retain as many original
    # counterfactual directions as possible. The final rank stays unchanged.
    _append_orthogonal(columns, attack_direction)
    _append_orthogonal(columns, mean_difference)

    for index in range(counterfactual_basis.shape[1]):
        if len(columns) >= rank:
            break
        _append_orthogonal(columns, counterfactual_basis[:, index])

    if len(columns) < rank:
        raise RuntimeError(
            f"Could construct only {len(columns)} independent directions for rank={rank}."
        )

    return np.column_stack(columns[:rank]), float(attack_model.score(standardized, labels))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shadow-updates",
        required=True,
        help="Associated shadow updates containing both updates and labels.",
    )
    parser.add_argument(
        "--counterfactual-basis",
        required=True,
        help="Existing counterfactual basis, for example final_r5.npz.",
    )
    parser.add_argument("--rank", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="results/defense_seed42_assoc07/subspace/final_attack_aware_r5.npz",
    )
    args = parser.parse_args()

    updates, labels = _load_updates_and_labels(args.shadow_updates)
    original = CounterfactualSubspace.load(args.counterfactual_basis)

    if updates.shape[1] != original.basis.shape[0]:
        raise ValueError(
            "Update dimension and basis dimension do not match: "
            f"updates={updates.shape}, basis={original.basis.shape}"
        )
    if args.rank < 1 or args.rank > original.basis.shape[0]:
        raise ValueError(f"Invalid rank={args.rank}")

    basis, training_attack_accuracy = build_attack_aware_basis(
        updates=updates,
        labels=labels,
        counterfactual_basis=original.basis,
        rank=args.rank,
        seed=args.seed,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        basis=basis,
        singular_values=original.singular_values,
        mean_difference=original.mean_difference,
        rank=np.asarray([args.rank], dtype=np.int64),
        basis_type="attack_aware_counterfactual",
        shadow_attack_training_accuracy=np.asarray(
            [training_attack_accuracy], dtype=np.float64
        ),
        seed=np.asarray([args.seed], dtype=np.int64),
    )

    gram_error = float(np.linalg.norm(basis.T @ basis - np.eye(args.rank)))
    print(f"Saved attack-aware basis: {output}")
    print(f"Basis shape: {basis.shape}")
    print(f"Shadow linear-attack training accuracy: {training_attack_accuracy:.6f}")
    print(f"Orthogonality error: {gram_error:.6e}")
    print("This is an empirical attack-aware defense, not a formal DP mechanism.")


if __name__ == "__main__":
    main()
