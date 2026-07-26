from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path
from typing import List, Mapping

import pandas as pd
import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}.")
    return data


def save_yaml(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def deep_set(mapping: dict, dotted_key: str, value) -> None:
    cursor = mapping
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def deep_get(mapping: Mapping, dotted_key: str, default=None):
    cursor = mapping
    for part in dotted_key.split("."):
        if not isinstance(cursor, Mapping) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor


def _merge_dict(base: dict, overlay: Mapping) -> dict:
    out = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _merge_dict(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _apply_dotted_overlay(cfg: dict, overlay: Mapping) -> dict:
    out = copy.deepcopy(cfg)
    for key, value in overlay.items():
        if "." in key:
            deep_set(out, key, value)
        elif isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _merge_dict(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _finalize_job(cfg: dict, profile: dict, factors: dict) -> dict:
    aggregation = str(deep_get(cfg, "federated.aggregation", "fedavg")).lower()
    if aggregation == "fedavg":
        deep_set(cfg, "federated.fedprox_mu", 0.0)
    elif aggregation == "fedprox" and float(deep_get(cfg, "federated.fedprox_mu", 0.0)) <= 0:
        deep_set(cfg, "federated.fedprox_mu", float(profile.get("default_fedprox_mu", 0.001)))
    cfg["experiment"]["profile_factors"] = factors
    return cfg


def expand_profile(matrix_cfg: dict, profile_name: str) -> List[dict]:
    """Expand one named E1-E11 profile into resolved jobs.

    `vary` creates a Cartesian product for independent factors. `cases` is a
    list of paired overlays used when settings must change together, e.g.
    architecture + tokenizer, or client count + samples/client.
    """
    datasets = matrix_cfg["datasets"]
    profile = matrix_cfg["profiles"][profile_name]
    dataset_names = profile.get("datasets", list(datasets))
    vary = profile.get("vary", {})
    vary_keys = list(vary)
    vary_values = [vary[key] for key in vary_keys]
    fixed = profile.get("fixed", {})
    cases = profile.get("cases", [None])
    jobs = []

    for dataset_name in dataset_names:
        base_path = datasets[dataset_name]
        base = load_yaml(base_path)
        base.setdefault("data", {})["name"] = dataset_name
        base.setdefault("experiment", {})["revision_experiment"] = profile_name
        base = _merge_dict(base, fixed)
        combinations = list(itertools.product(*vary_values)) if vary_keys else [tuple()]
        for case in cases:
            case_cfg = _apply_dotted_overlay(base, case or {})
            for values in combinations:
                cfg = copy.deepcopy(case_cfg)
                factors = {}
                if case:
                    factors.update({f"case::{k}": v for k, v in case.items()})
                for key, value in zip(vary_keys, values):
                    deep_set(cfg, key, value)
                    factors[key] = value
                jobs.append(_finalize_job(cfg, profile, factors))
    return jobs


def generate_job_files(matrix_path: str | Path, profile_name: str, output_dir: str | Path) -> pd.DataFrame:
    matrix_cfg = load_yaml(matrix_path)
    jobs = expand_profile(matrix_cfg, profile_name)
    output_dir = Path(output_dir) / profile_name
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, cfg in enumerate(jobs):
        job_id = f"{profile_name}_{idx:06d}"
        cfg["experiment"]["job_id"] = job_id
        path = output_dir / f"{job_id}.yaml"
        save_yaml(cfg, path)
        rows.append(
            {
                "job_id": job_id,
                "profile": profile_name,
                "config_path": str(path),
                "dataset": cfg["data"].get("name"),
                "seed": cfg.get("seed"),
                "population": cfg["experiment"].get("population", "target"),
                "setting_name": cfg["experiment"].get("setting_name"),
                "concentration": cfg["experiment"].get("concentration"),
                "architecture": cfg["model"].get("architecture"),
                "num_clients": cfg["federated"].get("num_clients"),
                "samples_per_client": cfg["federated"].get("samples_per_client"),
                "participation_rate": cfg["federated"].get("participation_rate"),
                "aggregation": cfg["federated"].get("aggregation"),
                "optimizer": cfg["federated"].get("optimizer"),
                "local_epochs": cfg["federated"].get("local_epochs"),
                "batch_size": cfg["federated"].get("batch_size"),
                "fusion_style": cfg["model"].get("fusion_style"),
                "fusion_stage": cfg["model"].get("fusion_stage"),
                "missing_mode": cfg["model"].get("missing_mode"),
                "defense": cfg.get("defense", {}).get("name", "none"),
                "defense_group": cfg.get("defense", {}).get("group"),
                "defense_rank": cfg.get("defense", {}).get("rank"),
                "defense_alpha": cfg.get("defense", {}).get("alpha"),
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_dir / "jobs.csv", index=False)
    return manifest


def select_defense_operating_point(
    validation_results_csv: str | Path,
    output_json: str | Path,
    attack_metric: str = "attack_asr",
    utility_metric: str = "task_auroc",
    max_utility_drop: float = 0.02,
    baseline_defense_name: str = "none",
) -> dict:
    """Select defense hyperparameters on shadow-validation only."""
    frame = pd.read_csv(validation_results_csv)
    baseline = frame[frame["defense_name"] == baseline_defense_name]
    if baseline.empty:
        raise ValueError("Validation results must include a no-defense baseline.")
    baseline_utility = float(baseline[utility_metric].mean())
    threshold = baseline_utility - float(max_utility_drop)
    feasible = frame[frame[utility_metric] >= threshold].copy()
    if feasible.empty:
        raise ValueError("No defense candidate satisfies the validation utility constraint.")
    selected = feasible.sort_values([attack_metric, utility_metric], ascending=[True, False]).iloc[0]
    result = {
        "selection_source": str(validation_results_csv),
        "selection_population": "shadow_val",
        "attack_metric": attack_metric,
        "utility_metric": utility_metric,
        "baseline_utility": baseline_utility,
        "max_utility_drop": max_utility_drop,
        "selected": selected.to_dict(),
        "target_results_used_for_selection": False,
    }
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def summarize_job_status(job_manifest: str | Path, results_root: str | Path) -> pd.DataFrame:
    jobs = pd.read_csv(job_manifest)
    root = Path(results_root)
    summaries = list(root.rglob("summary.json"))
    rows = []
    for row in jobs.to_dict("records"):
        completed = False
        summary_path = None
        for path in summaries:
            try:
                with path.open("r", encoding="utf-8") as f:
                    summary = json.load(f)
                if summary.get("seed") == row.get("seed") and summary.get("setting_name") == row.get("setting_name"):
                    completed = True
                    summary_path = str(path)
                    break
            except Exception:
                continue
        rows.append({**row, "completed": completed, "summary_path": summary_path})
    return pd.DataFrame(rows)
