from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd


def plot_concentration_leakage(
    csv_path: str | Path,
    output_path: str | Path,
    metric: str = "attack_asr",
    regime_column: str = "setting_name",
    concentration_column: str = "concentration",
    protocol: Optional[str] = "cross_partition_temporal",
) -> None:
    frame = pd.read_csv(csv_path)
    if protocol is not None and "protocol" in frame.columns:
        frame = frame[frame["protocol"] == protocol]
    fig, ax = plt.subplots(figsize=(7, 5))
    for regime, subset in frame.groupby(regime_column):
        grouped = subset.groupby(concentration_column)[metric].agg(["mean", "sem"]).reset_index()
        ax.errorbar(grouped[concentration_column], grouped["mean"], yerr=1.96 * grouped["sem"], marker="o", label=str(regime))
    ax.set_xlabel("Dominant-label concentration")
    ax.set_ylabel(metric)
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_privacy_utility(
    csv_path: str | Path,
    output_path: str | Path,
    privacy_metric: str = "attack_asr",
    utility_metric: str = "task_auroc",
    defense_column: str = "defense_name",
) -> None:
    frame = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(7, 5))
    for defense, subset in frame.groupby(defense_column):
        ax.scatter(subset[utility_metric], subset[privacy_metric], label=str(defense), alpha=0.75)
    ax.set_xlabel(utility_metric)
    ax.set_ylabel(privacy_metric)
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_layer_leakage(
    csv_path: str | Path,
    output_path: str | Path,
    metric: str = "attack_asr",
    group_column: str = "group",
) -> None:
    frame = pd.read_csv(csv_path)
    grouped = frame.groupby(group_column)[metric].agg(["mean", "sem"]).sort_values("mean")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(grouped.index.astype(str), grouped["mean"], xerr=1.96 * grouped["sem"])
    ax.set_xlabel(metric)
    ax.set_ylabel("Observed update component")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
