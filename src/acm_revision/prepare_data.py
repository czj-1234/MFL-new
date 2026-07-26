from __future__ import annotations

import argparse
import json

from .data_protocol import load_json, save_strict_pools
from .orchestrator import load_yaml
from .strict_split import build_strict_pools


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare strict ACM-revision pools without deleting original benchmark examples."
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = cfg["data"]
    split_cfg = cfg.get("strict_split", {})

    train = load_json(data_cfg["train_json"])
    val = load_json(data_cfg["val_json"])
    test = load_json(data_cfg["test_json"])

    pools, report = build_strict_pools(
        train,
        val,
        test,
        shadow_train_ratio=float(split_cfg.get("shadow_train_ratio", 0.4)),
        shadow_val_ratio=float(split_cfg.get("shadow_val_ratio", 0.2)),
        target_ratio=float(split_cfg.get("target_ratio", 0.4)),
        seed=int(cfg["seed"]),
        group_key=split_cfg.get("group_key"),
        group_near_duplicates=bool(split_cfg.get("group_near_duplicates", True)),
        text_hamming_threshold=int(split_cfg.get("text_hamming_threshold", 3)),
        image_hamming_threshold=int(split_cfg.get("image_hamming_threshold", 4)),
        min_text_tokens_for_near=int(split_cfg.get("min_text_tokens_for_near", 3)),
    )
    save_strict_pools(pools, report, data_cfg["pools_dir"])
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
