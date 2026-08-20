from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .attacks import AttackSplit, add_control_targets, build_attack_model, classification_metrics, load_matrix
from .defenses import name_in_group


def _target(frame: pd.DataFrame, target: str) -> np.ndarray:
    aliases = {"modality": "modality_target", "client_id": "client_identity", "round": "round_group"}
    column = aliases.get(target, target)
    if column not in frame.columns:
        raise KeyError(f"Missing target column {column}")
    return frame[column].to_numpy()


@lru_cache(maxsize=512)
def _checkpoint_statistics(run_dir: str, round_id: int) -> tuple[float, float, float]:
    path = Path(run_dir) / "checkpoints" / f"round_{int(round_id):04d}.pt"
    if not path.exists():
        return (float("nan"), float("nan"), float("nan"))
    state = torch.load(path, map_location="cpu")
    sums = {"full_update": 0.0, "classifier_head": 0.0, "fusion": 0.0}
    for name, tensor in state.items():
        if not torch.is_floating_point(tensor):
            continue
        sq = float(torch.sum(tensor.detach().cpu().float() ** 2).item())
        sums["full_update"] += sq
        if name_in_group(name, "classifier_head"):
            sums["classifier_head"] += sq
        if name_in_group(name, "fusion"):
            sums["fusion"] += sq
    return tuple(float(np.sqrt(sums[key])) for key in ("full_update", "classifier_head", "fusion"))


def _known_server_features(frame: pd.DataFrame, include_client_identity: bool = False, include_checkpoint: bool = True) -> np.ndarray:
    prepared = add_control_targets(frame)
    continuous = []
    for column in ("round", "num_samples", "participation_rate", "num_participants", "local_epochs", "batch_size"):
        if column in prepared.columns:
            values = pd.to_numeric(prepared[column], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            continuous.append(values[:, None])
    if include_checkpoint:
        checkpoint_rows = []
        for row in prepared.itertuples():
            update_path = Path(str(row.update_path))
            run_dir = str(update_path.parent.parent)
            checkpoint_rows.append(_checkpoint_statistics(run_dir, int(row.round)))
        continuous.append(np.asarray(checkpoint_rows, dtype=np.float32))

    categorical_cols = []
    if "modality" in prepared.columns:
        categorical_cols.append("modality")
    if "architecture" in prepared.columns:
        categorical_cols.append("architecture")
    if "aggregation" in prepared.columns:
        categorical_cols.append("aggregation")
    if include_client_identity and "client_key" in prepared.columns:
        categorical_cols.append("client_key")
    if categorical_cols:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        categorical = encoder.fit_transform(prepared[categorical_cols].astype(str)).astype(np.float32)
        continuous.append(categorical)
    if not continuous:
        return np.zeros((len(frame), 0), dtype=np.float32)
    return np.concatenate(continuous, axis=1)


def _aligned_server_features(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    include_client_identity: bool,
    include_checkpoint: bool,
):
    all_frame = pd.concat([train.assign(_split="train"), val.assign(_split="val"), test.assign(_split="test")], ignore_index=True)
    prepared = add_control_targets(all_frame)
    continuous_cols = [c for c in ("round", "num_samples", "participation_rate", "num_participants", "local_epochs", "batch_size") if c in prepared]
    continuous = prepared[continuous_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32) if continuous_cols else np.zeros((len(prepared), 0), dtype=np.float32)

    checkpoint = np.zeros((len(prepared), 0), dtype=np.float32)
    if include_checkpoint:
        values = []
        for row in prepared.itertuples():
            update_path = Path(str(row.update_path))
            values.append(_checkpoint_statistics(str(update_path.parent.parent), int(row.round)))
        checkpoint = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0)

    categorical_cols = [c for c in ("modality", "architecture", "aggregation") if c in prepared]
    if include_client_identity and "client_key" in prepared:
        categorical_cols.append("client_key")
    categorical = np.zeros((len(prepared), 0), dtype=np.float32)
    if categorical_cols:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        categorical = encoder.fit_transform(prepared[categorical_cols].astype(str)).astype(np.float32)

    features = np.concatenate([continuous, checkpoint, categorical], axis=1)
    n_train, n_val = len(train), len(val)
    return features[:n_train], features[n_train:n_train + n_val], features[n_train + n_val:]


def run_informed_server_attack(
    split: AttackSplit,
    group: str,
    attack_model: str = "mlp",
    target: str = "dominant_label",
    observation: str = "observed",
    seed: int = 42,
    pca_dim: Optional[int] = 256,
    include_client_identity: bool = False,
    include_checkpoint: bool = True,
) -> dict:
    """Attack using update features plus information the threat model grants the server."""
    train = add_control_targets(split.train, random_seed=seed)
    val = add_control_targets(split.val, random_seed=seed)
    test = add_control_targets(split.test, random_seed=seed)
    x_train = load_matrix(train, group, observation)
    x_val = load_matrix(val, group, observation)
    x_test = load_matrix(test, group, observation)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)
    if pca_dim and x_train.shape[1] > pca_dim:
        dim = min(int(pca_dim), x_train.shape[0] - 1, x_train.shape[1])
        pca = PCA(n_components=dim, random_state=seed)
        x_train = pca.fit_transform(x_train)
        x_val = pca.transform(x_val)
        x_test = pca.transform(x_test)

    m_train, m_val, m_test = _aligned_server_features(train, val, test, include_client_identity, include_checkpoint)
    m_scaler = StandardScaler()
    m_train = m_scaler.fit_transform(m_train) if m_train.shape[1] else m_train
    m_val = m_scaler.transform(m_val) if m_val.shape[1] else m_val
    m_test = m_scaler.transform(m_test) if m_test.shape[1] else m_test
    x_train = np.concatenate([x_train, m_train], axis=1)
    x_val = np.concatenate([x_val, m_val], axis=1)
    x_test = np.concatenate([x_test, m_test], axis=1)

    y_train = _target(train, target)
    y_test = _target(test, target)
    model = build_attack_model(attack_model, seed)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    score = model.predict_proba(x_test) if hasattr(model, "predict_proba") else (model.decision_function(x_test) if hasattr(model, "decision_function") else None)
    metrics = classification_metrics(y_test, pred, score)
    metrics.update(
        {
            "protocol": split.protocol,
            "group": group,
            "attack_model": f"informed_{attack_model}",
            "target": target,
            "observation": observation,
            "include_client_identity": include_client_identity,
            "include_checkpoint_statistics": include_checkpoint,
            "known_server_feature_dim": int(m_train.shape[1]),
            "n_train": len(train),
            "n_test": len(test),
        }
    )
    return metrics


def run_multi_round_majority_vote(
    split: AttackSplit,
    group: str,
    base_attack_model: str = "logistic_regression",
    target: str = "dominant_label",
    observation: str = "observed",
    seed: int = 42,
    pca_dim: Optional[int] = 256,
) -> dict:
    """Train a single-round attacker, then majority-vote its predictions per client trajectory."""
    train = add_control_targets(split.train, random_seed=seed)
    val = add_control_targets(split.val, random_seed=seed)
    test = add_control_targets(split.test, random_seed=seed)
    x_train = load_matrix(train, group, observation)
    x_val = load_matrix(val, group, observation)
    x_test = load_matrix(test, group, observation)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)
    if pca_dim and x_train.shape[1] > pca_dim:
        dim = min(int(pca_dim), x_train.shape[0] - 1, x_train.shape[1])
        pca = PCA(n_components=dim, random_state=seed)
        x_train = pca.fit_transform(x_train)
        x_val = pca.transform(x_val)
        x_test = pca.transform(x_test)

    y_train = _target(train, target)
    model = build_attack_model(base_attack_model, seed)
    model.fit(x_train, y_train)
    round_pred = model.predict(x_test)
    tmp = test[["client_key"]].copy()
    tmp["prediction"] = round_pred
    tmp["truth"] = _target(test, target)
    client_true, client_pred = [], []
    for _, client in tmp.groupby("client_key"):
        pred_values, pred_counts = np.unique(client["prediction"].to_numpy(), return_counts=True)
        true_values, true_counts = np.unique(client["truth"].to_numpy(), return_counts=True)
        client_pred.append(pred_values[pred_counts.argmax()])
        client_true.append(true_values[true_counts.argmax()])
    metrics = classification_metrics(np.asarray(client_true), np.asarray(client_pred))
    metrics.update(
        {
            "protocol": f"client_vote_{split.protocol}",
            "group": group,
            "attack_model": f"majority_vote_{base_attack_model}",
            "target": target,
            "observation": observation,
            "n_test_trajectories": len(client_true),
        }
    )
    return metrics
