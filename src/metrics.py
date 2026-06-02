# ============================================================
# Structure and Privacy Leakage Metrics
# Attackers: Random Forest + MLP + XGBoost
# ============================================================

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    accuracy_score,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from scipy.optimize import linear_sum_assignment

from xgboost import XGBClassifier


def effective_rank_from_singular_values(s):
    """
    Compute effective rank from singular values.
    """
    s = np.asarray(s, dtype=np.float64)
    s = s[s > 1e-12]

    if len(s) == 0:
        return 0.0

    p = s / s.sum()
    entropy = -np.sum(p * np.log(p + 1e-12))

    return float(np.exp(entropy))


def topk_energy_ratios(s, ks=(1, 3, 5)):
    """
    Compute Top-k singular value energy ratios.
    """
    s = np.asarray(s, dtype=np.float64)
    energy = s ** 2
    total = energy.sum() + 1e-12

    out = {}

    for k in ks:
        out[f"Top{k}_ratio"] = float(energy[:k].sum() / total)

    return out


def clustering_acc(y_true, y_pred):
    """
    Compute clustering accuracy using Hungarian matching.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    labels_true = np.unique(y_true)
    labels_pred = np.unique(y_pred)

    n = max(len(labels_true), len(labels_pred))
    cost = np.zeros((n, n), dtype=np.int64)

    true_to_idx = {label: i for i, label in enumerate(labels_true)}
    pred_to_idx = {label: i for i, label in enumerate(labels_pred)}

    for t, p in zip(y_true, y_pred):
        cost[true_to_idx[t], pred_to_idx[p]] += 1

    row_ind, col_ind = linear_sum_assignment(cost.max() - cost)
    matched = cost[row_ind, col_ind].sum()

    return matched / len(y_true)


def compute_attack_success_rates(X, y, seed=42):
    """
    Compute label-inference attack success rates.

    Attackers:
        1. Random Forest
        2. MLP
        3. XGBoost

    The purpose is to test whether label leakage is stable across
    different nonlinear attackers, rather than being caused by one
    specific classifier.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if len(np.unique(y)) < 2:
        return {
            "attack_success_rate_rf": np.nan,
            "attack_success_rate_mlp": np.nan,
            "attack_success_rate_xgb": np.nan,
            "attack_success_rate_mean": np.nan,
        }

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=seed,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    attackers = {
        "rf": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        ),

        "mlp": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            max_iter=500,
            random_state=seed,
            early_stopping=True,
        ),

        "xgb": XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softmax",
            num_class=len(np.unique(y)),
            eval_metric="mlogloss",
            random_state=seed,
            n_jobs=-1,
        ),
    }

    results = {}

    for name, attacker in attackers.items():
        try:
            attacker.fit(X_train_scaled, y_train)
            pred = attacker.predict(X_test_scaled)
            acc = accuracy_score(y_test, pred)

            results[f"attack_success_rate_{name}"] = float(acc)

        except Exception as e:
            print(f"[Warning] Attack model {name} failed: {e}")
            results[f"attack_success_rate_{name}"] = np.nan

    valid_scores = [
        v for v in results.values()
        if not np.isnan(v)
    ]

    results["attack_success_rate_mean"] = (
        float(np.mean(valid_scores)) if len(valid_scores) > 0 else np.nan
    )

    return results


def compute_structure_metrics(update_records, seed=42):
    """
    Compute structure and leakage metrics from update records.

    update_records should contain:
        {
            "update": np.ndarray,
            "dominant_label": int
        }
    """
    X = np.stack([r["update"] for r in update_records], axis=0)
    y = np.array([r["dominant_label"] for r in update_records])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    _, s, _ = np.linalg.svd(X_scaled, full_matrices=False)

    metrics = {}

    # -----------------------------
    # Structure metrics
    # -----------------------------
    metrics["e_rank"] = effective_rank_from_singular_values(s)
    metrics.update(topk_energy_ratios(s, ks=(1, 3, 5)))

    n_samples = X_scaled.shape[0]
    n_clusters = len(np.unique(y))

    if n_clusters >= n_samples:
        metrics["kmeans_acc"] = np.nan
        metrics["Silhouette"] = np.nan
        metrics["DBI"] = np.nan
        metrics["CHI"] = np.nan

    else:
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=seed,
            n_init=10,
        )

        pred_cluster = kmeans.fit_predict(X_scaled)

        metrics["kmeans_acc"] = clustering_acc(y, pred_cluster)

        pred_cluster_count = len(np.unique(pred_cluster))

        if 2 <= pred_cluster_count <= n_samples - 1:
            metrics["Silhouette"] = float(silhouette_score(X_scaled, pred_cluster))
            metrics["DBI"] = float(davies_bouldin_score(X_scaled, pred_cluster))
            metrics["CHI"] = float(calinski_harabasz_score(X_scaled, pred_cluster))
        else:
            metrics["Silhouette"] = np.nan
            metrics["DBI"] = np.nan
            metrics["CHI"] = np.nan

    # -----------------------------
    # Label leakage attack metrics
    # -----------------------------
    attack_metrics = compute_attack_success_rates(
        X_scaled,
        y,
        seed=seed,
    )

    metrics.update(attack_metrics)

    return metrics, X_scaled