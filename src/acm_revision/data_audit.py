from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

from .data_protocol import load_json, sample_identity


NATURAL_GROUP_CANDIDATES = (
    "client_id",
    "user_id",
    "author",
    "author_id",
    "source",
    "source_id",
    "topic",
    "topic_id",
    "collection",
    "collection_period",
    "community",
)


def _ids(data: Sequence[dict]) -> set[str]:
    return {sample_identity(item, idx) for idx, item in enumerate(data)}


def audit_dataset_files(
    train_json: str | Path,
    val_json: str | Path,
    test_json: str | Path,
    output_json: Optional[str | Path] = None,
) -> dict:
    train = load_json(train_json)
    val = load_json(val_json)
    test = load_json(test_json)
    ids = {"train": _ids(train), "val": _ids(val), "test": _ids(test)}
    overlaps = {
        "train_val": len(ids["train"] & ids["val"]),
        "train_test": len(ids["train"] & ids["test"]),
        "val_test": len(ids["val"] & ids["test"]),
    }

    available_natural_keys = {}
    for key in NATURAL_GROUP_CANDIDATES:
        values = [str(item[key]) for item in train if key in item and item[key] not in (None, "")]
        if values:
            counts = Counter(values)
            available_natural_keys[key] = {
                "coverage": len(values) / max(1, len(train)),
                "num_groups": len(counts),
                "min_group_size": min(counts.values()),
                "median_group_size": sorted(counts.values())[len(counts) // 2],
                "max_group_size": max(counts.values()),
            }

    result = {
        "sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "exact_identity_overlap": overlaps,
        "natural_partition_candidates": available_natural_keys,
        "natural_partition_available": bool(available_natural_keys),
        "note": (
            "Use a natural client partition only when a semantically valid source/user/author grouping exists. "
            "Do not invent pseudo-natural client IDs merely to satisfy an ablation."
        ),
    }
    if output_json:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    return result
