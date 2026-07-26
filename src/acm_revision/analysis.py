from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import ttest_rel, wilcoxon

from .attacks import load_matrix, read_metadata
from .defenses import name_in_group
from .fl import build_runtime, make_loader, resolve_label


def effective_rank(matrix: np.ndarray) -> float:
    singular = np.linalg.svd(matrix, compute_uv=False)
    p = singular / (singular.sum() + 1e-12)
    entropy = -np.sum(p * np.log(p + 1e-12))
    return float(np.exp(entropy))


def svd_energy(matrix: np.ndarray, ks: Sequence[int] = (1, 3, 5, 10)) -> dict:
    singular = np.linalg.svd(matrix, compute_uv=False)
    energy = singular ** 2
    total = energy.sum() + 1e-12
    out = {"effective_rank": effective_rank(matrix)}
    for k in ks:
        out[f"top{k}_energy"] = float(energy[: min(k, len(energy))].sum() / total)
    return out


def pairwise_cosine_summary(x: np.ndarray, labels: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    sim = x @ x.T
    within = []
    between = []
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            (within if labels[i] == labels[j] else between).append(float(sim[i, j]))
    return {
        "within_role_cosine_mean": float(np.mean(within)) if within else float("nan"),
        "between_role_cosine_mean": float(np.mean(between)) if between else float("nan"),
        "within_role_distance_mean": float(np.mean([1 - v for v in within])) if within else float("nan"),
        "between_role_distance_mean": float(np.mean([1 - v for v in between])) if between else float("nan"),
        "separation_gap": float(np.mean(within) - np.mean(between)) if within and between else float("nan"),
    }


def layer_geometry_report(
    metadata_paths: Sequence[str | Path],
    groups: Sequence[str],
    observation: str = "observed",
    output_csv: Optional[str | Path] = None,
) -> pd.DataFrame:
    frame = read_metadata(metadata_paths)
    rows = []
    group_by = ["setting_name", "concentration", "architecture"]
    for keys, subset in frame.groupby(group_by, dropna=False):
        for group in groups:
            try:
                x = load_matrix(subset, group, observation)
                labels = subset["dominant_label"].to_numpy()
                norms = np.linalg.norm(x, axis=1)
                row = {
                    **dict(zip(group_by, keys if isinstance(keys, tuple) else (keys,))),
                    "group": group,
                    "observation": observation,
                    "n_updates": len(subset),
                    "update_norm_mean": float(norms.mean()),
                    "update_norm_std": float(norms.std(ddof=1)) if len(norms) > 1 else 0.0,
                    **svd_energy(x),
                    **pairwise_cosine_summary(x, labels),
                }
                rows.append(row)
            except Exception as exc:
                rows.append(
                    {
                        **dict(zip(group_by, keys if isinstance(keys, tuple) else (keys,))),
                        "group": group,
                        "observation": observation,
                        "error": repr(exc),
                    }
                )
    result = pd.DataFrame(rows)
    if output_csv:
        path = Path(output_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(path, index=False)
    return result


def image_text_gradient_alignment(
    metadata_paths: Sequence[str | Path],
    group: str,
    observation: str = "observed",
) -> pd.DataFrame:
    frame = read_metadata(metadata_paths)
    rows = []
    grouping = ["run_id", "round", "dominant_label"]
    for keys, subset in frame.groupby(grouping):
        image_rows = subset[subset["modality"] == "image"]
        text_rows = subset[subset["modality"] == "text"]
        if image_rows.empty or text_rows.empty:
            continue
        image_mean = load_matrix(image_rows, group, observation).mean(axis=0)
        text_mean = load_matrix(text_rows, group, observation).mean(axis=0)
        cosine = float(
            np.dot(image_mean, text_mean)
            / ((np.linalg.norm(image_mean) + 1e-12) * (np.linalg.norm(text_mean) + 1e-12))
        )
        rows.append(
            {
                "run_id": keys[0],
                "round": keys[1],
                "dominant_label": keys[2],
                "group": group,
                "image_text_update_cosine": cosine,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_ci(
    frame: pd.DataFrame,
    value_column: str,
    unit_column: str,
    n_bootstrap: int = 5000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    units = frame[unit_column].dropna().unique().tolist()
    if not units:
        raise ValueError(f"No units found in {unit_column}.")
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(units, size=len(units), replace=True)
        pieces = [frame[frame[unit_column] == unit] for unit in sampled]
        boot = pd.concat(pieces, ignore_index=True)
        estimates.append(float(boot[value_column].mean()))
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(frame[value_column].mean()),
        "ci_low": float(np.quantile(estimates, alpha)),
        "ci_high": float(np.quantile(estimates, 1.0 - alpha)),
        "confidence": confidence,
        "n_units": len(units),
        "unit": unit_column,
    }


def paired_regime_test(
    frame: pd.DataFrame,
    value_column: str,
    regime_column: str,
    left_regime: str,
    right_regime: str,
    pair_columns: Sequence[str] = ("seed", "concentration", "num_clients", "participation_rate"),
) -> dict:
    left = frame[frame[regime_column] == left_regime]
    right = frame[frame[regime_column] == right_regime]
    left = left[list(pair_columns) + [value_column]].rename(columns={value_column: "left"})
    right = right[list(pair_columns) + [value_column]].rename(columns={value_column: "right"})
    merged = left.merge(right, on=list(pair_columns), how="inner").dropna()
    if len(merged) < 2:
        raise ValueError("At least two paired observations are required.")
    t = ttest_rel(merged["left"], merged["right"])
    try:
        w = wilcoxon(merged["left"], merged["right"])
        wilcoxon_stat, wilcoxon_p = float(w.statistic), float(w.pvalue)
    except Exception:
        wilcoxon_stat, wilcoxon_p = float("nan"), float("nan")
    diff = merged["left"] - merged["right"]
    return {
        "left_regime": left_regime,
        "right_regime": right_regime,
        "value_column": value_column,
        "n_pairs": int(len(merged)),
        "mean_paired_difference": float(diff.mean()),
        "ttest_statistic": float(t.statistic),
        "ttest_pvalue": float(t.pvalue),
        "wilcoxon_statistic": wilcoxon_stat,
        "wilcoxon_pvalue": wilcoxon_p,
    }


def aggregate_independent_runs(
    frame: pd.DataFrame,
    metric_columns: Sequence[str],
    group_columns: Sequence[str],
) -> pd.DataFrame:
    rows = []
    for keys, subset in frame.groupby(list(group_columns), dropna=False):
        row = dict(zip(group_columns, keys if isinstance(keys, tuple) else (keys,)))
        for metric in metric_columns:
            values = subset[metric].dropna().to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values)) if len(values) else float("nan")
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
            if len(values) > 1:
                rng = np.random.default_rng(42)
                boot = [float(np.mean(rng.choice(values, len(values), replace=True))) for _ in range(5000)]
                row[f"{metric}_ci_low"] = float(np.quantile(boot, 0.025))
                row[f"{metric}_ci_high"] = float(np.quantile(boot, 0.975))
            else:
                row[f"{metric}_ci_low"] = float("nan")
                row[f"{metric}_ci_high"] = float("nan")
        row["n_independent_runs"] = int(subset["seed"].nunique()) if "seed" in subset.columns else len(subset)
        rows.append(row)
    return pd.DataFrame(rows)


def privacy_utility_table(
    attack_csv: str | Path,
    run_summaries: Sequence[str | Path],
    output_csv: Optional[str | Path] = None,
) -> pd.DataFrame:
    attack = pd.read_csv(attack_csv)
    summaries = []
    for path in run_summaries:
        with open(path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        best = summary.get("best") or {}
        summaries.append(
            {
                "run_id": summary["run_id"],
                "test_acc": best.get("test_acc"),
                "test_macro_f1": best.get("test_macro_f1"),
                "test_auroc": best.get("test_auroc"),
                "defense_name": (summary.get("defense") or {}).get("name", "none"),
            }
        )
    utility = pd.DataFrame(summaries)
    if "run_id" in attack.columns:
        merged = attack.merge(utility, on="run_id", how="left")
    else:
        merged = attack.assign(_key=1).merge(utility.assign(_key=1), on="_key").drop(columns="_key")
    if output_csv:
        path = Path(output_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(path, index=False)
    return merged


def class_conditional_gradient_probe(
    cfg: dict,
    checkpoint_path: str | Path,
    data: Sequence[dict],
    group: str,
    mode: str,
    label_source: str = "auto",
    samples_per_class: int = 32,
    seed: int = 42,
) -> dict:
    """Measure class-conditional gradient directions at one fixed checkpoint."""
    rng = np.random.default_rng(seed)
    num_classes = int(cfg["data"]["num_classes"])
    model, tokenizer, image_processor = build_runtime(cfg)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=True)
    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    gradients = {}

    for label in range(num_classes):
        candidates = [x for x in data if resolve_label(x, mode, label_source) == label]
        if len(candidates) < samples_per_class:
            raise ValueError(f"Need {samples_per_class} samples for class {label}, found {len(candidates)}.")
        indices = rng.choice(len(candidates), size=samples_per_class, replace=False)
        subset = [candidates[int(i)] for i in indices]
        loader = make_loader(
            subset,
            tokenizer,
            image_processor,
            mode,
            cfg,
            shuffle=False,
            seed=seed,
            label_source=label_source,
            batch_size=samples_per_class,
        )
        batch = next(iter(loader))
        model.zero_grad(set_to_none=True)
        logits = model(
            pixel_values=batch["pixel_values"].to(device),
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            setting=mode,
        )
        labels = batch["label"].to(device)
        F.cross_entropy(logits, labels).backward()
        parts = []
        for name, param in model.named_parameters():
            if name_in_group(name, group) and param.grad is not None:
                parts.append(param.grad.detach().cpu().reshape(-1).numpy())
        if not parts:
            raise ValueError(f"No gradients matched group={group}.")
        gradients[label] = np.concatenate(parts).astype(np.float32)

    cosine = np.full((num_classes, num_classes), np.nan, dtype=float)
    for i in range(num_classes):
        for j in range(num_classes):
            a, b = gradients[i], gradients[j]
            cosine[i, j] = float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)))
    return {
        "group": group,
        "num_classes": num_classes,
        "samples_per_class": samples_per_class,
        "gradient_norms": {str(k): float(np.linalg.norm(v)) for k, v in gradients.items()},
        "class_conditional_gradient_cosine": cosine.tolist(),
    }
