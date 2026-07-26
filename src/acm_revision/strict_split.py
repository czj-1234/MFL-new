from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .data_protocol import (
    StrictPools,
    _dhash64,
    _hamming64,
    _normalise_text,
    _sample_image_path,
    _sample_text,
    _simhash64,
    sample_identity,
    validate_pool_disjointness,
)


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def _annotate_unique_ids(data: Sequence[dict]) -> Tuple[List[dict], dict]:
    """Attach a unique strict ID without deleting any original example.

    Explicit duplicate IDs are preserved as separate rows but are also assigned the
    same `_strict_base_identity`, so they can be kept in the same split.
    """
    out: List[dict] = []
    occurrences: Counter[str] = Counter()
    duplicate_base_ids = 0

    for idx, item in enumerate(data):
        x = dict(item)
        base = sample_identity(x, idx)
        occurrence = occurrences[base]
        occurrences[base] += 1
        if occurrence > 0:
            duplicate_base_ids += 1
        strict_id = base if occurrence == 0 else f"{base}::occurrence={occurrence}"
        x["_strict_base_identity"] = base
        x["_strict_sample_id"] = strict_id
        out.append(x)

    return out, {
        "input_examples": len(data),
        "retained_examples": len(out),
        "removed_examples": 0,
        "duplicate_base_identity_rows": duplicate_base_ids,
        "policy": "preserve_all_rows",
    }


def _token_count(text: str) -> int:
    return len([t for t in _normalise_text(text).split() if t])


def build_duplicate_families(
    data: Sequence[dict],
    *,
    group_key: Optional[str] = None,
    group_near_duplicates: bool = True,
    text_hamming_threshold: int = 3,
    image_hamming_threshold: int = 4,
    min_text_tokens_for_near: int = 3,
) -> Tuple[List[List[int]], dict]:
    """Build split-level duplicate/source families without removing examples.

    Why group instead of delete:
      * the original benchmark remains intact;
      * the same raw example is never reused across pools;
      * exact/repeated text and near-image/near-text families cannot straddle
        Shadow-Train, Shadow-Val and Target, avoiding attack train/test leakage.

    The near-duplicate search is deliberately conservative and bucketed. It is a
    split guard, not a claim that two benchmark examples are semantically identical.
    """
    n = len(data)
    uf = _UnionFind(n)
    stats = Counter()

    # Exact/base identity duplicates.
    by_base: Dict[str, int] = {}
    for idx, item in enumerate(data):
        base = str(item.get("_strict_base_identity") or sample_identity(item, idx))
        if base in by_base:
            if uf.union(idx, by_base[base]):
                stats["base_identity_links"] += 1
        else:
            by_base[base] = idx

    # Optional source/topic/author/collection grouping.
    if group_key:
        first_by_source: Dict[str, int] = {}
        for idx, item in enumerate(data):
            value = item.get(group_key)
            if value in (None, ""):
                continue
            key = str(value)
            if key in first_by_source:
                if uf.union(idx, first_by_source[key]):
                    stats["source_group_links"] += 1
            else:
                first_by_source[key] = idx

    # Exact normalized text: keep repeated templates in the same split, but retain
    # every paired meme example.
    first_by_text: Dict[str, int] = {}
    for idx, item in enumerate(data):
        text = _normalise_text(_sample_text(item))
        if not text:
            continue
        if text in first_by_text:
            if uf.union(idx, first_by_text[text]):
                stats["exact_text_links"] += 1
        else:
            first_by_text[text] = idx

    if group_near_duplicates:
        text_buckets: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
        image_buckets: Dict[int, List[Tuple[int, int]]] = defaultdict(list)

        for idx, item in enumerate(data):
            text = _sample_text(item)
            norm_text = _normalise_text(text)
            token_count = _token_count(text)
            if norm_text and token_count >= min_text_tokens_for_near:
                text_hash = _simhash64(text)
                bucket = (text_hash >> 48) & 0xFFFF
                for other_idx, other_hash, other_tokens in text_buckets[bucket]:
                    if min(token_count, other_tokens) < min_text_tokens_for_near:
                        continue
                    if _hamming64(text_hash, other_hash) <= text_hamming_threshold:
                        if uf.union(idx, other_idx):
                            stats["near_text_links"] += 1
                text_buckets[bucket].append((idx, text_hash, token_count))

            image_hash = _dhash64(_sample_image_path(item))
            if image_hash is not None:
                bucket = (image_hash >> 48) & 0xFFFF
                for other_idx, other_hash in image_buckets[bucket]:
                    if _hamming64(image_hash, other_hash) <= image_hamming_threshold:
                        if uf.union(idx, other_idx):
                            stats["near_image_links"] += 1
                image_buckets[bucket].append((idx, image_hash))

    grouped: Dict[int, List[int]] = defaultdict(list)
    for idx in range(n):
        grouped[uf.find(idx)].append(idx)

    families = list(grouped.values())
    sizes = [len(x) for x in families]
    duplicate_sizes = [s for s in sizes if s > 1]
    report = {
        "mode": "group_without_deletion",
        "group_near_duplicates": bool(group_near_duplicates),
        "text_hamming_threshold": int(text_hamming_threshold),
        "image_hamming_threshold": int(image_hamming_threshold),
        "min_text_tokens_for_near": int(min_text_tokens_for_near),
        "num_families": len(families),
        "num_multi_example_families": len(duplicate_sizes),
        "largest_family_size": max(sizes) if sizes else 0,
        "examples_in_multi_example_families": int(sum(duplicate_sizes)),
        **{k: int(v) for k, v in stats.items()},
    }
    return families, report


def _assign_families_to_pools(
    data: Sequence[dict],
    families: Sequence[Sequence[int]],
    ratios: Sequence[float],
    seed: int,
) -> Tuple[List[List[dict]], dict]:
    rng = random.Random(seed)
    family_order = list(range(len(families)))
    rng.shuffle(family_order)
    # Large groups first reduces ratio drift while still randomizing equal-size groups.
    family_order.sort(key=lambda i: len(families[i]), reverse=True)

    total = len(data)
    targets = [float(r) * total for r in ratios]
    buckets: List[List[dict]] = [[], [], []]
    bucket_family_ids: List[List[int]] = [[], [], []]

    for family_id in family_order:
        members = list(families[family_id])
        deficits = [targets[i] - len(buckets[i]) for i in range(3)]
        # Prefer the pool with the greatest remaining deficit. Once all are full,
        # choose the smallest relative overshoot.
        bucket_id = int(np.argmax(deficits))
        for idx in members:
            x = dict(data[idx])
            x["_strict_duplicate_group"] = int(family_id)
            buckets[bucket_id].append(x)
        bucket_family_ids[bucket_id].append(int(family_id))

    report = {
        "target_sizes": {
            "shadow_train": targets[0],
            "shadow_val": targets[1],
            "target": targets[2],
        },
        "actual_sizes": {
            "shadow_train": len(buckets[0]),
            "shadow_val": len(buckets[1]),
            "target": len(buckets[2]),
        },
        "family_counts": {
            "shadow_train": len(bucket_family_ids[0]),
            "shadow_val": len(bucket_family_ids[1]),
            "target": len(bucket_family_ids[2]),
        },
    }
    return buckets, report


def _validate_duplicate_groups_do_not_cross(pools: StrictPools) -> dict:
    names = ("shadow_train", "shadow_val", "target")
    groups = {
        name: {int(x["_strict_duplicate_group"]) for x in getattr(pools, name)}
        for name in names
    }
    report = {}
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = groups[left].intersection(groups[right])
            report[f"{left}__{right}"] = len(overlap)
            if overlap:
                raise ValueError(
                    f"Duplicate/source families crossed strict pools: {left} vs {right}: {len(overlap)} groups"
                )
    return report


def _identity_set(data: Sequence[Mapping]) -> set[str]:
    return {sample_identity(x, i) for i, x in enumerate(data)}


def _audit_official_split_identity_overlap(pools: StrictPools) -> dict:
    all_sets = {
        "shadow_train": {str(x["_strict_base_identity"]) for x in pools.shadow_train},
        "shadow_val": {str(x["_strict_base_identity"]) for x in pools.shadow_val},
        "target": {str(x["_strict_base_identity"]) for x in pools.target},
        "task_val": _identity_set(pools.task_val),
        "task_test": _identity_set(pools.task_test),
    }
    report = {}
    names = list(all_sets)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            count = len(all_sets[left].intersection(all_sets[right]))
            report[f"{left}__{right}"] = count
    # Any exact identity leakage between client-building pools and official task
    # evaluation sets is a data-integrity error. Val-vs-test overlap is also flagged.
    bad = {k: v for k, v in report.items() if v > 0}
    if bad:
        raise ValueError(f"Exact sample identity overlap across official/strict splits: {bad}")
    return report


def build_strict_pools(
    train_data: Sequence[dict],
    task_val: Sequence[dict],
    task_test: Sequence[dict],
    shadow_train_ratio: float = 0.40,
    shadow_val_ratio: float = 0.20,
    target_ratio: float = 0.40,
    seed: int = 42,
    group_key: Optional[str] = None,
    group_near_duplicates: bool = True,
    text_hamming_threshold: int = 3,
    image_hamming_threshold: int = 4,
    min_text_tokens_for_near: int = 3,
) -> Tuple[StrictPools, dict]:
    """Create strict pools while preserving the original benchmark examples.

    No global near-duplicate deletion is performed. Instead, duplicate/source
    families are assigned as indivisible split units so they cannot leak from
    Shadow to Target. The official validation/test sets remain untouched.
    """
    ratios = np.asarray([shadow_train_ratio, shadow_val_ratio, target_ratio], dtype=float)
    if np.any(ratios < 0) or not np.isclose(ratios.sum(), 1.0):
        raise ValueError("shadow_train_ratio + shadow_val_ratio + target_ratio must equal 1.")

    working, preservation_report = _annotate_unique_ids(train_data)
    families, duplicate_report = build_duplicate_families(
        working,
        group_key=group_key,
        group_near_duplicates=group_near_duplicates,
        text_hamming_threshold=text_hamming_threshold,
        image_hamming_threshold=image_hamming_threshold,
        min_text_tokens_for_near=min_text_tokens_for_near,
    )
    buckets, assignment_report = _assign_families_to_pools(working, families, ratios, seed)

    pools = StrictPools(
        shadow_train=buckets[0],
        shadow_val=buckets[1],
        target=buckets[2],
        task_val=[dict(x) for x in task_val],
        task_test=[dict(x) for x in task_test],
    )

    strict_id_overlap = validate_pool_disjointness(pools)
    duplicate_group_overlap = _validate_duplicate_groups_do_not_cross(pools)
    official_identity_overlap = _audit_official_split_identity_overlap(pools)

    report = {
        "seed": int(seed),
        "group_key": group_key,
        "ratios": {
            "shadow_train": float(shadow_train_ratio),
            "shadow_val": float(shadow_val_ratio),
            "target": float(target_ratio),
        },
        "sizes": {
            "shadow_train": len(pools.shadow_train),
            "shadow_val": len(pools.shadow_val),
            "target": len(pools.target),
            "task_val": len(pools.task_val),
            "task_test": len(pools.task_test),
        },
        "preservation": preservation_report,
        "duplicate_guard": duplicate_report,
        "assignment": assignment_report,
        "strict_sample_id_overlap": strict_id_overlap,
        "duplicate_group_overlap": duplicate_group_overlap,
        "official_exact_identity_overlap": official_identity_overlap,
    }
    return pools, report
