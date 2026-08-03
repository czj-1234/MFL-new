from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from . import fl
from .defenses import flatten_delta, name_in_group
from .orchestrator import load_yaml


DEFAULT_EVERY_ROUND_GROUPS = ["classifier_bias", "classifier_weight", "classifier_head"]
DEFAULT_MILESTONE_EXACT_GROUPS = ["fusion", "missing_modality"]
DEFAULT_MILESTONE_SKETCH_GROUPS = ["image_encoder", "text_encoder", "all_shared", "full_update"]
DEFAULT_MILESTONE_ROUNDS = [1, 5, 10, 20, 30, 50, 75, 100, 125, 150]


def _stable_seed(*parts: object) -> int:
    text = "|".join(str(x) for x in parts)
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") & 0x7FFFFFFF


def _round_from_path(path: Path) -> int:
    match = re.search(r"round_(\d+)_client_", path.name)
    if not match:
        raise ValueError(f"Could not parse round from update path: {path}")
    return int(match.group(1))


def _matching_layout(delta: Mapping[str, torch.Tensor], group: str):
    names = []
    sizes = []
    for name in sorted(delta):
        tensor = delta[name]
        if name_in_group(name, group):
            names.append(name)
            sizes.append(int(tensor.numel()))
    return names, sizes


def _coordinate_sketch_pair(
    raw_delta: Mapping[str, torch.Tensor],
    observed_delta: Mapping[str, torch.Tensor],
    group: str,
    sketch_dim: int,
    seed: int,
):
    names, sizes = _matching_layout(observed_delta, group)
    original_dim = int(sum(sizes))
    if original_dim <= 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32), 0

    k = min(int(sketch_dim), original_dim)
    rng = np.random.default_rng(_stable_seed(seed, group, original_dim))
    sampled = np.sort(rng.choice(original_dim, size=k, replace=False).astype(np.int64))
    raw_out = np.empty(k, dtype=np.float32)
    obs_out = np.empty(k, dtype=np.float32)

    offsets = np.cumsum(np.asarray([0] + sizes, dtype=np.int64))
    cursor = 0
    for idx, name in enumerate(names):
        left, right = int(offsets[idx]), int(offsets[idx + 1])
        end = int(np.searchsorted(sampled, right, side="left"))
        if end <= cursor:
            continue
        local = sampled[cursor:end] - left
        raw_flat = raw_delta[name].detach().cpu().numpy().reshape(-1)
        obs_flat = observed_delta[name].detach().cpu().numpy().reshape(-1)
        raw_out[cursor:end] = raw_flat[local]
        obs_out[cursor:end] = obs_flat[local]
        cursor = end

    if cursor != k:
        raise AssertionError(f"Sketch extraction incomplete for {group}: {cursor}/{k}")
    return raw_out, obs_out, original_dim


def _save_capture_npz_factory(cfg: dict):
    capture_cfg = cfg.get("privacy_capture", {})
    every_groups = list(capture_cfg.get("every_round_exact_groups", DEFAULT_EVERY_ROUND_GROUPS))
    milestone_exact = list(capture_cfg.get("milestone_exact_groups", DEFAULT_MILESTONE_EXACT_GROUPS))
    milestone_sketch = list(capture_cfg.get("milestone_sketch_groups", DEFAULT_MILESTONE_SKETCH_GROUPS))
    milestone_rounds = {int(x) for x in capture_cfg.get("milestone_rounds", DEFAULT_MILESTONE_ROUNDS)}
    sketch_dim = int(capture_cfg.get("sketch_dim", 16384))
    sketch_seed = int(capture_cfg.get("sketch_seed", 20260803))

    def save_capture_npz(
        path: Path,
        raw_delta: Mapping[str, torch.Tensor],
        observed_delta: Mapping[str, torch.Tensor],
        groups: Sequence[str],
        storage_dtype: str,
    ) -> dict:
        del groups
        round_id = _round_from_path(path)
        exact_groups = list(every_groups)
        sketch_groups = []
        if round_id in milestone_rounds:
            exact_groups.extend(g for g in milestone_exact if g not in exact_groups)
            sketch_groups = list(milestone_sketch)

        dtype = np.float16 if storage_dtype == "float16" else np.float32
        arrays: Dict[str, np.ndarray] = {}
        dimensions: Dict[str, int] = {}
        representations: Dict[str, dict] = {}

        for group in exact_groups:
            raw, _ = flatten_delta(raw_delta, group)
            observed, _ = flatten_delta(observed_delta, group)
            if observed.size == 0:
                continue
            arrays[f"raw__{group}"] = raw.astype(dtype, copy=False)
            arrays[f"observed__{group}"] = observed.astype(dtype, copy=False)
            dimensions[group] = int(observed.size)
            representations[group] = {
                "representation": "exact",
                "feature_dim": int(observed.size),
                "original_dim": int(observed.size),
            }

        for group in sketch_groups:
            raw, observed, original_dim = _coordinate_sketch_pair(
                raw_delta,
                observed_delta,
                group,
                sketch_dim=sketch_dim,
                seed=sketch_seed,
            )
            if observed.size == 0:
                continue
            arrays[f"raw__{group}"] = raw.astype(dtype, copy=False)
            arrays[f"observed__{group}"] = observed.astype(dtype, copy=False)
            dimensions[group] = int(observed.size)
            representations[group] = {
                "representation": "deterministic_coordinate_sketch",
                "feature_dim": int(observed.size),
                "original_dim": int(original_dim),
                "sketch_seed": sketch_seed,
            }

        if not arrays:
            raise RuntimeError(f"No update arrays selected for {path}")

        capture_info = {
            "round": round_id,
            "milestone": round_id in milestone_rounds,
            "groups": representations,
            "storage_dtype": storage_dtype,
        }
        arrays["__capture_info_json"] = np.frombuffer(
            json.dumps(capture_info, sort_keys=True).encode("utf-8"), dtype=np.uint8
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
        return dimensions

    return save_capture_npz


def _read_capture_info(update_path: str | Path) -> dict:
    with np.load(update_path, allow_pickle=False) as data:
        if "__capture_info_json" not in data.files:
            raise KeyError(f"Missing __capture_info_json in {update_path}")
        raw = bytes(np.asarray(data["__capture_info_json"], dtype=np.uint8).tolist())
        return json.loads(raw.decode("utf-8"))


def enrich_and_validate_run(run_dir: str | Path, cfg: dict | None = None) -> dict:
    run_dir = Path(run_dir)
    metadata_path = run_dir / "update_metadata.csv"
    summary_path = run_dir / "summary.json"
    if not metadata_path.exists() or not summary_path.exists():
        raise FileNotFoundError(f"Missing metadata/summary under {run_dir}")

    frame = pd.read_csv(metadata_path)
    if frame.empty:
        raise ValueError(f"Empty update metadata: {metadata_path}")

    groups = [
        "classifier_bias",
        "classifier_weight",
        "classifier_head",
        "fusion",
        "image_encoder",
        "text_encoder",
        "missing_modality",
        "all_shared",
        "full_update",
    ]
    info_cache: Dict[str, dict] = {}
    for path in frame["update_path"].astype(str).unique():
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Missing update file: {p}")
        if p.stat().st_size <= 128:
            raise ValueError(f"Update file is unexpectedly small: {p}")
        info_cache[path] = _read_capture_info(p)

    for group in groups:
        frame[f"has_{group}"] = frame["update_path"].astype(str).map(
            lambda p: group in info_cache[p].get("groups", {})
        )
        frame[f"repr_{group}"] = frame["update_path"].astype(str).map(
            lambda p: (info_cache[p].get("groups", {}).get(group) or {}).get("representation")
        )
        frame[f"original_dim_{group}"] = frame["update_path"].astype(str).map(
            lambda p: (info_cache[p].get("groups", {}).get(group) or {}).get("original_dim")
        )

    capture_cfg = (cfg or {}).get("privacy_capture", {})
    milestone_rounds = {
        int(x) for x in capture_cfg.get("milestone_rounds", DEFAULT_MILESTONE_ROUNDS)
    }
    observed_rounds = set(frame["round"].astype(int).unique())
    expected_milestones = sorted(observed_rounds & milestone_rounds)

    every_round_groups = list(
        capture_cfg.get("every_round_exact_groups", DEFAULT_EVERY_ROUND_GROUPS)
    )
    milestone_groups = list(
        capture_cfg.get("milestone_exact_groups", DEFAULT_MILESTONE_EXACT_GROUPS)
    ) + list(capture_cfg.get("milestone_sketch_groups", DEFAULT_MILESTONE_SKETCH_GROUPS))

    failures = []
    for group in every_round_groups:
        if not bool(frame[f"has_{group}"].all()):
            failures.append(f"{group} is not present on every update")
    for group in milestone_groups:
        subset = frame[frame["round"].astype(int).isin(expected_milestones)]
        if not subset.empty and not bool(subset[f"has_{group}"].all()):
            failures.append(f"{group} is missing on one or more milestone updates")

    frame.to_csv(metadata_path, index=False)
    manifest = {
        "run_dir": str(run_dir),
        "n_updates": int(len(frame)),
        "rounds": sorted(int(x) for x in observed_rounds),
        "milestone_rounds_present": expected_milestones,
        "groups": {
            group: {
                "n_available": int(frame[f"has_{group}"].sum()),
                "representations": sorted(
                    str(x) for x in frame[f"repr_{group}"].dropna().unique().tolist()
                ),
            }
            for group in groups
        },
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
    }
    with (run_dir / "capture_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    if failures:
        raise RuntimeError("; ".join(failures))
    return manifest


def run_from_config(config_path: str, population: str | None = None) -> dict:
    cfg = load_yaml(config_path)
    population = population or cfg.get("experiment", {}).get("population", "target")
    cfg = copy.deepcopy(cfg)

    capture_cfg = cfg.setdefault("privacy_capture", {})
    capture_cfg.setdefault("every_round_exact_groups", DEFAULT_EVERY_ROUND_GROUPS)
    capture_cfg.setdefault("milestone_exact_groups", DEFAULT_MILESTONE_EXACT_GROUPS)
    capture_cfg.setdefault("milestone_sketch_groups", DEFAULT_MILESTONE_SKETCH_GROUPS)
    capture_cfg.setdefault("milestone_rounds", DEFAULT_MILESTONE_ROUNDS)
    capture_cfg.setdefault("sketch_dim", 16384)
    capture_cfg.setdefault("sketch_seed", 20260803)

    all_groups = []
    for key in ("every_round_exact_groups", "milestone_exact_groups", "milestone_sketch_groups"):
        for group in capture_cfg[key]:
            if group not in all_groups:
                all_groups.append(group)
    cfg.setdefault("update_capture", {})["groups"] = all_groups
    cfg["update_capture"]["save_all_checkpoints"] = False
    cfg["update_capture"]["checkpoint_rounds"] = list(
        cfg["update_capture"].get("model_checkpoint_rounds", [])
    )

    original_saver = fl._save_update_npz
    fl._save_update_npz = _save_capture_npz_factory(cfg)
    try:
        summary = fl.run_strict_fl(cfg, population=population)
    finally:
        fl._save_update_npz = original_saver

    run_dir = Path(cfg["experiment"].get("output_root", "results/acm_revision")) / summary["run_id"]
    manifest = enrich_and_validate_run(run_dir, cfg)
    summary["capture_manifest"] = manifest
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Round-aware privacy update capture runner")
    parser.add_argument("--config")
    parser.add_argument("--population", choices=["shadow_train", "shadow_val", "target"])
    parser.add_argument("--validate-run-dir")
    args = parser.parse_args()

    if args.validate_run_dir:
        cfg = load_yaml(args.config) if args.config else None
        print(json.dumps(enrich_and_validate_run(args.validate_run_dir, cfg), ensure_ascii=False, indent=2))
        return
    if not args.config:
        raise SystemExit("--config is required unless --validate-run-dir is used")
    print(json.dumps(run_from_config(args.config, args.population), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
