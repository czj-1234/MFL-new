import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.defense.subspace import CounterfactualSubspace


def _load_updates_and_labels(path, num_clients=4):
    data = np.load(path)
    updates = data["updates"] if "updates" in data.files else data[data.files[0]]
    updates = np.asarray(updates, dtype=np.float64)

    if updates.ndim != 2:
        raise ValueError(f"Expected a 2D update matrix, got {updates.shape}")

    if "labels" in data.files:
        labels = np.asarray(data["labels"], dtype=np.int64)
        label_source = "stored_in_npz"
    else:
        if num_clients <= 0:
            raise ValueError("num_clients must be positive")
        if updates.shape[0] % num_clients != 0:
            raise ValueError(
                "Cannot infer labels because the number of updates is not divisible "
                f"by num_clients: updates={updates.shape[0]}, num_clients={num_clients}."
            )

        # In the current four-client modality-exclusive design, updates are saved
        # round by round in client order 0,1,2,3. Dominant labels are assigned by
        # get_client_dominant_label(client_id) = client_id % 2, giving 0,1,0,1.
        client_labels = np.arange(num_clients, dtype=np.int64) % 2
        labels = np.tile(client_labels, updates.shape[0] // num_clients)
        label_source = f"inferred_from_client_order_{client_labels.tolist()}"

    if labels.ndim != 1 or labels.shape[0] != updates.shape[0]:
        raise ValueError(
            f"Label shape {labels.shape} is incompatible with updates {updates.shape}"
        )

    return updates, labels, label_source


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
        help="Associated shadow updates. Labels are used if present; otherwise they are inferred from client order.",
    )
    parser.add_argument(
        "--counterfactual-basis",
        required=True,
        help="Existing counterfactual basis, for example final_r5.npz.",
    )
    parser.add_argument("--rank", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-clients",
        type=int,
        default=4,
        help="Number of clients used to infer labels when the NPZ has no labels.",
    )
    parser.add_argument(
        "--output",
        default="results/defense_seed42_assoc07/subspace/final_attack_aware_r5.npz",
    )
    args = parser.parse_args()

    updates, labels, label_source = _load_updates_and_labels(
        args.shadow_updates,
        num_clients=args.num_clients,
    )
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
        label_source=np.asarray([label_source]),
    )

    gram_error = float(np.linalg.norm(basis.T @ basis - np.eye(args.rank)))
    counts = {int(label): int((labels == label).sum()) for label in np.unique(labels)}
    print(f"Loaded shadow updates: {updates.shape}")
    print(f"Label source: {label_source}")
    print(f"Label counts: {counts}")
    print(f"Saved attack-aware basis: {output}")
    print(f"Basis shape: {basis.shape}")
    print(f"Shadow linear-attack training accuracy: {training_attack_accuracy:.6f}")
    print(f"Orthogonality error: {gram_error:.6e}")
    print("This is an empirical attack-aware defense, not a formal DP mechanism.")


if __name__ == "__main__":
    main()
