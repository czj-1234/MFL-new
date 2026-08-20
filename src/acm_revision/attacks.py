from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier


@dataclass
class AttackSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    protocol: str


def read_metadata(paths: Sequence[str | Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["metadata_source"] = str(path)
        frames.append(frame)
    if not frames:
        raise ValueError("No metadata files were provided.")
    return pd.concat(frames, ignore_index=True)


def _load_vector(path: str, group: str, observation: str = "observed") -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        key = f"{observation}__{group}"
        if key not in data.files:
            raise KeyError(f"{key} missing from {path}; available={data.files}")
        return np.asarray(data[key], dtype=np.float32)


def load_matrix(frame: pd.DataFrame, group: str, observation: str = "observed") -> np.ndarray:
    vectors = [_load_vector(path, group, observation) for path in frame["update_path"].tolist()]
    dims = {v.size for v in vectors}
    if len(dims) != 1:
        raise ValueError(f"Inconsistent update dimensions: {sorted(dims)}")
    return np.stack(vectors, axis=0)


def add_control_targets(frame: pd.DataFrame, num_round_groups: int = 3, random_seed: int = 42) -> pd.DataFrame:
    out = frame.copy()
    max_round = max(1, int(out["round"].max()))
    bins = np.linspace(0, max_round + 1, num_round_groups + 1)
    out["round_group"] = np.digitize(out["round"].to_numpy(), bins[1:-1], right=True)
    out["client_identity"] = out["client_key"].astype("category").cat.codes
    out["modality_target"] = out["modality"].astype("category").cat.codes

    def random_group(key: str) -> int:
        digest = hashlib.blake2b(f"{random_seed}|{key}".encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "little") % 2

    out["random_group"] = out["client_key"].astype(str).map(random_group)
    return out


def _stratified_client_partition(frame: pd.DataFrame, seed: int, train_frac: float = 0.6, val_frac: float = 0.2):
    clients = frame[["client_key", "dominant_label"]].drop_duplicates()
    rng = random.Random(seed)
    by_label: Dict[int, List[str]] = {}
    for row in clients.itertuples():
        by_label.setdefault(int(row.dominant_label), []).append(str(row.client_key))
    train_clients, val_clients, test_clients = [], [], []
    for keys in by_label.values():
        rng.shuffle(keys)
        n = len(keys)
        n_train = max(1, int(round(n * train_frac))) if n >= 3 else max(1, n - 2)
        n_val = max(1, int(round(n * val_frac))) if n - n_train >= 2 else max(0, n - n_train - 1)
        train_clients += keys[:n_train]
        val_clients += keys[n_train : n_train + n_val]
        test_clients += keys[n_train + n_val :]
    if not test_clients:
        raise ValueError("Not enough clients for a client-disjoint test split.")
    return set(train_clients), set(val_clients), set(test_clients)


def make_attack_split(
    frame: pd.DataFrame,
    protocol: str,
    seed: int = 42,
    early_fraction: float = 0.4,
    late_fraction: float = 0.4,
) -> AttackSplit:
    frame = frame.copy().reset_index(drop=True)
    protocol = protocol.lower()
    rng = np.random.default_rng(seed)

    if protocol == "random_update":
        indices = np.arange(len(frame))
        rng.shuffle(indices)
        a = int(0.6 * len(indices))
        b = int(0.8 * len(indices))
        return AttackSplit(frame.iloc[indices[:a]], frame.iloc[indices[a:b]], frame.iloc[indices[b:]], protocol)

    if protocol == "temporal":
        rounds = sorted(frame["round"].unique().tolist())
        n = len(rounds)
        early_n = max(1, int(math.floor(n * early_fraction)))
        late_n = max(1, int(math.floor(n * late_fraction)))
        early = set(rounds[:early_n])
        late = set(rounds[-late_n:])
        middle = set(rounds) - early - late
        if not middle:
            middle = {rounds[min(early_n, n - 1)]}
        return AttackSplit(frame[frame["round"].isin(early)], frame[frame["round"].isin(middle)], frame[frame["round"].isin(late)], protocol)

    if protocol == "cross_client":
        train_clients, val_clients, test_clients = _stratified_client_partition(frame, seed)
        return AttackSplit(
            frame[frame["client_key"].isin(train_clients)],
            frame[frame["client_key"].isin(val_clients)],
            frame[frame["client_key"].isin(test_clients)],
            protocol,
        )

    if protocol == "cross_partition":
        required = {"shadow_train", "shadow_val", "target"}
        observed = set(frame["population"].astype(str).unique())
        if not required.issubset(observed):
            raise ValueError(f"cross_partition requires populations {sorted(required)}, observed {sorted(observed)}")
        return AttackSplit(
            frame[frame["population"] == "shadow_train"],
            frame[frame["population"] == "shadow_val"],
            frame[frame["population"] == "target"],
            protocol,
        )

    if protocol == "cross_partition_temporal":
        required = {"shadow_train", "shadow_val", "target"}
        observed = set(frame["population"].astype(str).unique())
        if not required.issubset(observed):
            raise ValueError(f"cross_partition_temporal requires populations {sorted(required)}")
        max_round = int(frame["round"].max())
        early_cut = max(1, int(math.floor(max_round * early_fraction)))
        late_cut = max(1, int(math.ceil(max_round * (1.0 - late_fraction))))
        train = frame[(frame["population"] == "shadow_train") & (frame["round"] <= early_cut)]
        val = frame[(frame["population"] == "shadow_val") & (frame["round"] > early_cut)]
        test = frame[(frame["population"] == "target") & (frame["round"] >= late_cut)]
        return AttackSplit(train, val, test, protocol)

    raise ValueError(f"Unknown attack protocol: {protocol}")


def _target_values(frame: pd.DataFrame, target: str) -> np.ndarray:
    aliases = {
        "modality": "modality_target",
        "client_id": "client_identity",
        "round": "round_group",
    }
    column = aliases.get(target, target)
    if column not in frame.columns:
        raise KeyError(f"Target column {column} is not available.")
    return frame[column].to_numpy()


def _compress_fit_transform(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    pca_dim: Optional[int],
    standardize: bool,
):
    scaler = None
    if standardize:
        scaler = StandardScaler(with_mean=True, with_std=True)
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)
        x_test = scaler.transform(x_test)
    pca = None
    if pca_dim is not None and pca_dim > 0 and x_train.shape[1] > pca_dim:
        max_dim = min(pca_dim, x_train.shape[0] - 1, x_train.shape[1])
        pca = PCA(n_components=max_dim, random_state=0)
        x_train = pca.fit_transform(x_train)
        x_val = pca.transform(x_val)
        x_test = pca.transform(x_test)
    return x_train, x_val, x_test, scaler, pca


def _scores(model, x: np.ndarray):
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x)
        return probs
    if hasattr(model, "decision_function"):
        scores = model.decision_function(x)
        if scores.ndim == 1:
            return scores
        return scores
    return None


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores=None) -> dict:
    out = {
        "attack_acc": float(accuracy_score(y_true, y_pred)),
        "attack_asr": float(accuracy_score(y_true, y_pred)),
        "attack_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "attack_balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
    }
    try:
        classes = np.unique(y_true)
        if scores is None:
            raise ValueError
        scores = np.asarray(scores)
        if len(classes) == 2:
            if scores.ndim == 2:
                scores = scores[:, 1]
            out["attack_auroc"] = float(roc_auc_score(y_true, scores))
        else:
            out["attack_auroc"] = float(roc_auc_score(y_true, scores, multi_class="ovr", average="macro"))
    except Exception:
        out["attack_auroc"] = float("nan")
    return out


def _centroid_predict(x_train: np.ndarray, y_train: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    classes = sorted(np.unique(y_train).tolist())
    centroids = []
    for cls in classes:
        c = x_train[y_train == cls].mean(axis=0)
        c = c / (np.linalg.norm(c) + 1e-12)
        centroids.append(c)
    cmat = np.stack(centroids)
    xnorm = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    similarity = xnorm @ cmat.T
    pred_idx = similarity.argmax(axis=1)
    pred = np.asarray([classes[i] for i in pred_idx])
    return pred, similarity


def build_attack_model(name: str, seed: int, params: Optional[dict] = None):
    params = dict(params or {})
    name = name.lower()
    if name == "logistic_regression":
        return LogisticRegression(max_iter=int(params.pop("max_iter", 3000)), random_state=seed, class_weight="balanced", **params)
    if name == "linear_svm":
        return LinearSVC(random_state=seed, class_weight="balanced", **params)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=int(params.pop("n_estimators", 500)), random_state=seed, class_weight="balanced", n_jobs=-1, **params)
    if name == "boosted_trees":
        return GradientBoostingClassifier(random_state=seed, **params)
    if name == "xgboost":
        return XGBClassifier(
            n_estimators=int(params.pop("n_estimators", 500)),
            max_depth=int(params.pop("max_depth", 5)),
            learning_rate=float(params.pop("learning_rate", 0.05)),
            subsample=float(params.pop("subsample", 0.9)),
            colsample_bytree=float(params.pop("colsample_bytree", 0.9)),
            random_state=seed,
            n_jobs=-1,
            eval_metric="mlogloss",
            **params,
        )
    if name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=tuple(params.pop("hidden_layer_sizes", [256, 128])),
            max_iter=int(params.pop("max_iter", 500)),
            early_stopping=True,
            random_state=seed,
            **params,
        )
    raise ValueError(f"Unknown attack model: {name}")


def run_tabular_attack(
    split: AttackSplit,
    group: str,
    attack_model: str,
    target: str = "dominant_label",
    observation: str = "observed",
    seed: int = 42,
    pca_dim: Optional[int] = 256,
    model_params: Optional[dict] = None,
) -> dict:
    train = add_control_targets(split.train, random_seed=seed)
    val = add_control_targets(split.val, random_seed=seed)
    test = add_control_targets(split.test, random_seed=seed)
    y_train = _target_values(train, target)
    y_val = _target_values(val, target)
    y_test = _target_values(test, target)

    x_train = load_matrix(train, group, observation)
    x_val = load_matrix(val, group, observation)
    x_test = load_matrix(test, group, observation)

    if attack_model == "majority":
        values, counts = np.unique(y_train, return_counts=True)
        pred = np.full_like(y_test, values[counts.argmax()])
        metrics = classification_metrics(y_test, pred)
    elif attack_model == "random":
        rng = np.random.default_rng(seed)
        classes = np.unique(y_train)
        pred = rng.choice(classes, size=len(y_test), replace=True)
        metrics = classification_metrics(y_test, pred)
    elif attack_model == "update_norm":
        xtr = np.linalg.norm(x_train, axis=1, keepdims=True)
        xte = np.linalg.norm(x_test, axis=1, keepdims=True)
        model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed).fit(xtr, y_train)
        pred = model.predict(xte)
        metrics = classification_metrics(y_test, pred, _scores(model, xte))
    elif attack_model == "cosine_centroid":
        pred, score = _centroid_predict(x_train, y_train, x_test)
        metrics = classification_metrics(y_test, pred, score)
    else:
        x_train, x_val, x_test, _, pca = _compress_fit_transform(x_train, x_val, x_test, pca_dim, standardize=True)
        model = build_attack_model(attack_model, seed, model_params)
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        metrics = classification_metrics(y_test, pred, _scores(model, x_test))
        metrics["attack_feature_dim"] = int(x_train.shape[1])
        metrics["pca_used"] = pca is not None

    metrics.update(
        {
            "protocol": split.protocol,
            "group": group,
            "observation": observation,
            "attack_model": attack_model,
            "target": target,
            "n_train": int(len(train)),
            "n_val": int(len(val)),
            "n_test": int(len(test)),
            "n_train_clients": int(train["client_key"].nunique()),
            "n_test_clients": int(test["client_key"].nunique()),
        }
    )
    return metrics


def run_label_distribution_inference(
    split: AttackSplit,
    group: str,
    num_classes: int,
    observation: str = "observed",
    seed: int = 42,
    pca_dim: Optional[int] = 256,
) -> dict:
    x_train = load_matrix(split.train, group, observation)
    x_val = load_matrix(split.val, group, observation)
    x_test = load_matrix(split.test, group, observation)
    x_train, x_val, x_test, _, _ = _compress_fit_transform(x_train, x_val, x_test, pca_dim, standardize=True)
    y_train = split.train[[f"label_prop_{i}" for i in range(num_classes)]].to_numpy(dtype=np.float32)
    y_test_dist = split.test[[f"label_prop_{i}" for i in range(num_classes)]].to_numpy(dtype=np.float32)
    true_dominant = y_test_dist.argmax(axis=1)

    models = []
    predictions = []
    for cls in range(num_classes):
        model = Ridge(alpha=1.0)
        model.fit(x_train, y_train[:, cls])
        predictions.append(model.predict(x_test))
        models.append(model)
    pred_dist = np.stack(predictions, axis=1)
    pred_dist = np.maximum(pred_dist, 0)
    pred_dist = pred_dist / (pred_dist.sum(axis=1, keepdims=True) + 1e-12)
    pred_dominant = pred_dist.argmax(axis=1)
    metrics = classification_metrics(true_dominant, pred_dominant, pred_dist)
    metrics.update(
        {
            "protocol": split.protocol,
            "group": group,
            "observation": observation,
            "attack_model": "label_distribution_ridge_argmax",
            "target": "dominant_label_via_distribution",
            "distribution_mae": float(np.mean(np.abs(pred_dist - y_test_dist))),
            "n_train": int(len(split.train)),
            "n_test": int(len(split.test)),
        }
    )
    return metrics


def build_trajectory_arrays(
    frame: pd.DataFrame,
    group: str,
    observation: str,
    target: str,
    sequence_length: Optional[int] = None,
) -> Tuple[List[np.ndarray], np.ndarray, List[str]]:
    prepared = add_control_targets(frame)
    sequences, labels, keys = [], [], []
    for client_key, group_frame in prepared.groupby("client_key"):
        group_frame = group_frame.sort_values("round")
        vectors = load_matrix(group_frame, group, observation)
        if sequence_length is not None:
            if len(vectors) < sequence_length:
                continue
            vectors = vectors[-sequence_length:]
        y = _target_values(group_frame, target)
        values, counts = np.unique(y, return_counts=True)
        sequences.append(vectors)
        labels.append(values[counts.argmax()])
        keys.append(str(client_key))
    return sequences, np.asarray(labels), keys


class SequenceClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, architecture: str):
        super().__init__()
        self.architecture = architecture
        if architecture == "lstm":
            self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.head = nn.Linear(hidden_dim, num_classes)
        elif architecture == "tcn":
            self.encoder = nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
            )
            self.head = nn.Linear(hidden_dim, num_classes)
        elif architecture == "transformer":
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)
            self.head = nn.Linear(hidden_dim, num_classes)
        else:
            raise ValueError(f"Unknown sequence architecture: {architecture}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.architecture == "lstm":
            _, (h, _) = self.encoder(x)
            feature = h[-1]
        elif self.architecture == "tcn":
            feature = self.encoder(x.transpose(1, 2)).mean(dim=2)
        else:
            feature = self.encoder(self.input_proj(x)).mean(dim=1)
        return self.head(feature)


def _trajectory_features(sequences: List[np.ndarray], method: str) -> np.ndarray:
    if method == "temporal_mean":
        return np.stack([s.mean(axis=0) for s in sequences])
    if method == "temporal_variance":
        return np.stack([s.var(axis=0) for s in sequences])
    if method == "mean_variance":
        return np.stack([np.concatenate([s.mean(axis=0), s.var(axis=0)]) for s in sequences])
    if method == "concatenation":
        lengths = {len(s) for s in sequences}
        if len(lengths) != 1:
            raise ValueError("concatenation requires equal trajectory lengths; set sequence_length.")
        return np.stack([s.reshape(-1) for s in sequences])
    raise ValueError(f"Unknown trajectory feature method: {method}")


def run_trajectory_statistical_attack(
    split: AttackSplit,
    group: str,
    method: str,
    attack_model: str = "logistic_regression",
    target: str = "dominant_label",
    observation: str = "observed",
    sequence_length: Optional[int] = None,
    pca_dim: Optional[int] = 256,
    seed: int = 42,
) -> dict:
    train_seq, y_train, _ = build_trajectory_arrays(split.train, group, observation, target, sequence_length)
    val_seq, y_val, _ = build_trajectory_arrays(split.val, group, observation, target, sequence_length)
    test_seq, y_test, _ = build_trajectory_arrays(split.test, group, observation, target, sequence_length)
    x_train = _trajectory_features(train_seq, method)
    x_val = _trajectory_features(val_seq, method)
    x_test = _trajectory_features(test_seq, method)
    x_train, x_val, x_test, _, _ = _compress_fit_transform(x_train, x_val, x_test, pca_dim, standardize=True)
    model = build_attack_model(attack_model, seed)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    metrics = classification_metrics(y_test, pred, _scores(model, x_test))
    metrics.update(
        {
            "protocol": f"trajectory_{split.protocol}",
            "trajectory_method": method,
            "attack_model": attack_model,
            "group": group,
            "target": target,
            "n_train_trajectories": len(train_seq),
            "n_test_trajectories": len(test_seq),
        }
    )
    return metrics


def _fit_update_pca(train_seq: List[np.ndarray], val_seq: List[np.ndarray], test_seq: List[np.ndarray], dim: int):
    train_updates = np.concatenate(train_seq, axis=0)
    max_dim = min(dim, train_updates.shape[0] - 1, train_updates.shape[1])
    pca = PCA(n_components=max_dim, random_state=0).fit(train_updates)
    transform = lambda seqs: [pca.transform(s).astype(np.float32) for s in seqs]
    return transform(train_seq), transform(val_seq), transform(test_seq), pca


def _pad_sequences(sequences: List[np.ndarray], length: int) -> np.ndarray:
    dim = sequences[0].shape[1]
    out = np.zeros((len(sequences), length, dim), dtype=np.float32)
    for i, sequence in enumerate(sequences):
        seq = sequence[-length:]
        out[i, -len(seq) :] = seq
    return out


def run_neural_trajectory_attack(
    split: AttackSplit,
    group: str,
    architecture: str,
    target: str = "dominant_label",
    observation: str = "observed",
    sequence_length: int = 10,
    update_pca_dim: int = 64,
    hidden_dim: int = 128,
    epochs: int = 100,
    lr: float = 1e-3,
    seed: int = 42,
    device: Optional[str] = None,
) -> dict:
    torch.manual_seed(seed)
    train_seq, y_train, _ = build_trajectory_arrays(split.train, group, observation, target, sequence_length)
    val_seq, y_val, _ = build_trajectory_arrays(split.val, group, observation, target, sequence_length)
    test_seq, y_test, _ = build_trajectory_arrays(split.test, group, observation, target, sequence_length)
    train_seq, val_seq, test_seq, _ = _fit_update_pca(train_seq, val_seq, test_seq, update_pca_dim)
    x_train = _pad_sequences(train_seq, sequence_length)
    x_val = _pad_sequences(val_seq, sequence_length)
    x_test = _pad_sequences(test_seq, sequence_length)

    classes = sorted(np.unique(y_train).tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_train_idx = np.asarray([class_to_idx[c] for c in y_train], dtype=np.int64)
    y_val_idx = np.asarray([class_to_idx[c] for c in y_val], dtype=np.int64)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = SequenceClassifier(x_train.shape[2], hidden_dim, len(classes), architecture).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    xt = torch.from_numpy(x_train).to(dev)
    yt = torch.from_numpy(y_train_idx).to(dev)
    xv = torch.from_numpy(x_val).to(dev)
    yv = torch.from_numpy(y_val_idx).to(dev)
    best_state = None
    best_loss = float("inf")
    patience = 15
    stale = 0

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(xt), yt)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(xv), yv).item()) if len(x_val) else float(loss.item())
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x_test).to(dev)).cpu().numpy()
    pred_idx = logits.argmax(axis=1)
    pred = np.asarray([classes[i] for i in pred_idx])
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    metrics = classification_metrics(y_test, pred, probs)
    metrics.update(
        {
            "protocol": f"trajectory_{split.protocol}",
            "trajectory_method": architecture,
            "attack_model": architecture,
            "group": group,
            "target": target,
            "n_train_trajectories": len(train_seq),
            "n_test_trajectories": len(test_seq),
            "update_pca_dim": int(x_train.shape[2]),
        }
    )
    return metrics


def run_attack_grid(
    metadata_paths: Sequence[str | Path],
    output_csv: str | Path,
    protocols: Sequence[str],
    groups: Sequence[str],
    models: Sequence[str],
    targets: Sequence[str],
    observation: str = "observed",
    seed: int = 42,
    pca_dim: Optional[int] = 256,
) -> pd.DataFrame:
    frame = add_control_targets(read_metadata(metadata_paths), random_seed=seed)
    rows = []
    for protocol in protocols:
        split = make_attack_split(frame, protocol, seed=seed)
        for group in groups:
            for target in targets:
                for model in models:
                    try:
                        rows.append(run_tabular_attack(split, group, model, target, observation, seed, pca_dim))
                    except Exception as exc:
                        rows.append(
                            {
                                "protocol": protocol,
                                "group": group,
                                "target": target,
                                "attack_model": model,
                                "observation": observation,
                                "error": repr(exc),
                            }
                        )
    result = pd.DataFrame(rows)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    return result
