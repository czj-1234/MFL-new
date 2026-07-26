from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler

from .attacks import AttackSplit, _scores, _target_values, add_control_targets, build_attack_model, classification_metrics, load_matrix


DEFAULT_GRIDS = {
    "logistic_regression": {
        "C": [0.01, 0.1, 1.0, 10.0, 100.0],
    },
    "linear_svm": {
        "C": [0.01, 0.1, 1.0, 10.0],
    },
    "random_forest": {
        "n_estimators": [300, 600],
        "max_depth": [None, 8, 16],
        "min_samples_leaf": [1, 3, 5],
    },
    "boosted_trees": {
        "n_estimators": [100, 300],
        "learning_rate": [0.03, 0.1],
        "max_depth": [2, 4],
    },
    "xgboost": {
        "n_estimators": [300, 600],
        "max_depth": [3, 5, 8],
        "learning_rate": [0.03, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    },
    "mlp": {
        "hidden_layer_sizes": [[128], [256, 128], [512, 256]],
        "alpha": [1e-5, 1e-4, 1e-3],
        "learning_rate_init": [1e-4, 1e-3],
        "max_iter": [800],
    },
}


def _preprocess(split: AttackSplit, group: str, observation: str, pca_dim: Optional[int]):
    train = add_control_targets(split.train)
    val = add_control_targets(split.val)
    test = add_control_targets(split.test)
    x_train = load_matrix(train, group, observation)
    x_val = load_matrix(val, group, observation)
    x_test = load_matrix(test, group, observation)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)
    pca = None
    if pca_dim and x_train.shape[1] > pca_dim:
        dim = min(int(pca_dim), x_train.shape[0] - 1, x_train.shape[1])
        pca = PCA(n_components=dim, random_state=42)
        x_train = pca.fit_transform(x_train)
        x_val = pca.transform(x_val)
        x_test = pca.transform(x_test)
    return train, val, test, x_train, x_val, x_test


def _selection_score(y_true, pred, scores, metric: str) -> float:
    if metric == "macro_f1":
        return float(f1_score(y_true, pred, average="macro", zero_division=0))
    if metric == "accuracy":
        return float(accuracy_score(y_true, pred))
    if metric == "auroc":
        try:
            scores = np.asarray(scores)
            classes = np.unique(y_true)
            if len(classes) == 2:
                if scores.ndim == 2:
                    scores = scores[:, 1]
                return float(roc_auc_score(y_true, scores))
            return float(roc_auc_score(y_true, scores, multi_class="ovr", average="macro"))
        except Exception:
            return float("-inf")
    raise ValueError("selection_metric must be macro_f1, accuracy, or auroc")


def tune_and_test_attack(
    split: AttackSplit,
    group: str,
    attack_model: str,
    target: str = "dominant_label",
    observation: str = "observed",
    seed: int = 42,
    pca_dim: Optional[int] = 256,
    parameter_grid: Optional[Dict] = None,
    selection_metric: str = "macro_f1",
    output_json: Optional[str | Path] = None,
) -> dict:
    """Tune only on attack-train/attack-validation and evaluate target once.

    The selected hyperparameters are frozen before any test prediction. The
    exact same function/grid should be used for every modality regime.
    """
    train, val, test, x_train, x_val, x_test = _preprocess(split, group, observation, pca_dim)
    y_train = _target_values(train, target)
    y_val = _target_values(val, target)
    y_test = _target_values(test, target)
    grid = parameter_grid if parameter_grid is not None else DEFAULT_GRIDS.get(attack_model, {})
    candidates = list(ParameterGrid(grid)) if grid else [{}]
    trials = []
    best_params = None
    best_score = float("-inf")

    for params in candidates:
        model = build_attack_model(attack_model, seed, params=params)
        model.fit(x_train, y_train)
        pred = model.predict(x_val)
        score_obj = _scores(model, x_val)
        score = _selection_score(y_val, pred, score_obj, selection_metric)
        trials.append({"params": params, "validation_score": score})
        if score > best_score:
            best_score = score
            best_params = dict(params)

    # Freeze the selected configuration. Refit on attack-train only by default
    # so the validation partition remains a true tuning set and is never mixed
    # with target/test data.
    model = build_attack_model(attack_model, seed, params=best_params)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    metrics = classification_metrics(y_test, pred, _scores(model, x_test))
    result = {
        **metrics,
        "protocol": split.protocol,
        "group": group,
        "observation": observation,
        "attack_model": attack_model,
        "target": target,
        "selection_metric": selection_metric,
        "best_validation_score": best_score,
        "best_params": best_params,
        "n_candidates": len(candidates),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "test_used_for_tuning": False,
        "trials": trials,
    }
    if output_json:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    return result
