from __future__ import annotations

import argparse
import json
from typing import List, Optional

from .data_protocol import load_json, save_strict_pools
from .orchestrator import load_yaml
from .strict_split import build_strict_pools


def _as_str_list(value) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value]
    raise ValueError(
        f"Expected stratify_keys to be a list or comma-separated string, got {type(value).__name__}"
    )


def prepare_from_config(config_path: str) -> dict:
    cfg = load_yaml(config_path)
    data_cfg = cfg["data"]
    split_cfg = cfg.get("strict_split", {})

    train = load_json(data_cfg["train_json"])
    val = load_json(data_cfg["val_json"])
    test = load_json(data_cfg["test_json"])

    pools, report = build_strict_pools(
        train,
        val,
        test,
        shadow_train_ratio=float(split_cfg.get("shadow_train_ratio", 0.40)),
        shadow_val_ratio=float(split_cfg.get("shadow_val_ratio", 0.20)),
        target_ratio=float(split_cfg.get("target_ratio", 0.40)),
        seed=int(cfg.get("seed", 42)),
        group_key=split_cfg.get("group_key"),
        exact_deduplicate=bool(split_cfg.get("exact_deduplicate", True)),
        group_near_duplicates=bool(split_cfg.get("group_near_duplicates", True)),
        text_hamming_threshold=int(split_cfg.get("text_hamming_threshold", 3)),
        image_hamming_threshold=int(split_cfg.get("image_hamming_threshold", 4)),
        min_text_tokens_for_near=int(split_cfg.get("min_text_tokens_for_near", 3)),
        stratify_keys=_as_str_list(split_cfg.get("stratify_keys")),
        size_weight=float(split_cfg.get("size_weight", 1.0)),
        stratify_weight=float(split_cfg.get("stratify_weight", 2.0)),
        refinement_passes=int(split_cfg.get("refinement_passes", 3)),
        swap_attempts_per_pass=int(split_cfg.get("swap_attempts_per_pass", 5000)),
    )
    save_strict_pools(pools, report, data_cfg["pools_dir"])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare strict ACM-revision pools with exact deduplication and group-aware stratification."
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    report = prepare_from_config(args.config)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
