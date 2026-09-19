from __future__ import annotations

import json
import random
from collections import Counter
from typing import List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .data_protocol import StrictPools, validate_pool_disjointness
from . import strict_split as base


POOL_NAMES = base.POOL_NAMES
DEFAULT_FAMILY_SIZE_BINS = (1, 2, 4, 8)


def _family_size_bin(size: int, boundaries: Sequence[int]) -> str:
    """Map family size to stable bins: 1, 2, 3-4, 5-8, 9+ by default."""
    if size <= 0:
        raise ValueError("family size must be positive")
    bounds = [int(x) for x in boundaries]
    if bounds != sorted(bounds) or len(set(bounds)) != len(bounds):
        raise ValueError("family_size_bins must be strictly increasing")
    lower = 1
    for upper in bounds:
        if size <= upper:
            return str(upper) if lower == upper else f"{lower}-{upper}"
        lower = upper + 1
    return f"{lower}+"


def _counter_add(target: Counter, source: Counter, sign: int = 1) -> None:
    for key, value in source.items():
        target[key] += sign * int(value)
        if target[key] == 0:
            del target[key]


def _relative_l1(actual: Mapping[str, int], target: Mapping[str, float], keys: Sequence[str], normalizer: float) -> float:
    if not keys:
        return 0.0
    error = sum(abs(float(actual.get(k, 0)) - float(target.get(k, 0.0))) for k in keys)
    return error / max(1.0, 2.0 * float(normalizer))


def _family_structure_report(
    families: Sequence[Sequence[int]],
    assignment: Sequence[int],
    ratios: Sequence[float],
    family_size_bins: Sequence[int],
) -> dict:
    global_family_counts = Counter()
    global_sample_counts = Counter()
    pool_family_counts = [Counter(), Counter(), Counter()]
    pool_sample_counts = [Counter(), Counter(), Counter()]

    for family_id, members in enumerate(families):
        size = len(members)
        bin_name = _family_size_bin(size, family_size_bins)
        global_family_counts[bin_name] += 1
        global_sample_counts[bin_name] += size
        pool_id = int(assignment[family_id])
        pool_family_counts[pool_id][bin_name] += 1
        pool_sample_counts[pool_id][bin_name] += size

    bins = sorted(global_sample_counts, key=lambda x: (0 if x.isdigit() else 1, x))
    total_samples = sum(global_sample_counts.values())
    multi_samples_global = sum(size for size in global_sample_counts.values()) - global_sample_counts.get("1", 0)

    pools = {}
    max_sample_bin_drift = 0.0
    max_family_bin_drift = 0.0
    multi_sample_fraction_values = []

    for pool_id, name in enumerate(POOL_NAMES):
        pool_samples = sum(pool_sample_counts[pool_id].values())
        pool_families = sum(pool_family_counts[pool_id].values())
        sample_props = {
            b: (pool_sample_counts[pool_id].get(b, 0) / pool_samples if pool_samples else 0.0)
            for b in bins
        }
        family_props = {
            b: (pool_family_counts[pool_id].get(b, 0) / pool_families if pool_families else 0.0)
            for b in bins
        }
        global_sample_props = {
            b: (global_sample_counts.get(b, 0) / total_samples if total_samples else 0.0)
            for b in bins
        }
        total_families = sum(global_family_counts.values())
        global_family_props = {
            b: (global_family_counts.get(b, 0) / total_families if total_families else 0.0)
            for b in bins
        }
        sample_drift = max((abs(sample_props[b] - global_sample_props[b]) for b in bins), default=0.0)
        family_drift = max((abs(family_props[b] - global_family_props[b]) for b in bins), default=0.0)
        max_sample_bin_drift = max(max_sample_bin_drift, sample_drift)
        max_family_bin_drift = max(max_family_bin_drift, family_drift)

        singleton_samples = pool_sample_counts[pool_id].get("1", 0)
        multi_fraction = (pool_samples - singleton_samples) / pool_samples if pool_samples else 0.0
        multi_sample_fraction_values.append(multi_fraction)
        pools[name] = {
            "num_samples": int(pool_samples),
            "num_families": int(pool_families),
            "family_count_by_size_bin": dict(sorted(pool_family_counts[pool_id].items())),
            "sample_count_by_family_size_bin": dict(sorted(pool_sample_counts[pool_id].items())),
            "family_proportion_by_size_bin": family_props,
            "sample_proportion_by_family_size_bin": sample_props,
            "multi_example_family_sample_fraction": float(multi_fraction),
            "max_sample_bin_proportion_drift": float(sample_drift),
            "max_family_bin_proportion_drift": float(family_drift),
        }

    return {
        "family_size_bins": [int(x) for x in family_size_bins],
        "global_family_count_by_size_bin": dict(sorted(global_family_counts.items())),
        "global_sample_count_by_family_size_bin": dict(sorted(global_sample_counts.items())),
        "global_multi_example_family_sample_fraction": (
            float(multi_samples_global / total_samples) if total_samples else 0.0
        ),
        "pools": pools,
        "max_sample_bin_proportion_drift": float(max_sample_bin_drift),
        "max_family_bin_proportion_drift": float(max_family_bin_drift),
        "multi_example_family_sample_fraction_range": (
            float(max(multi_sample_fraction_values) - min(multi_sample_fraction_values))
            if multi_sample_fraction_values
            else 0.0
        ),
    }


def _assign_families_to_pools(
    data: Sequence[dict],
    families: Sequence[Sequence[int]],
    ratios: Sequence[float],
    seed: int,
    *,
    stratify_keys: Optional[Sequence[str]] = None,
    size_weight: float = 1.0,
    stratify_weight: float = 2.0,
    family_structure_weight: float = 2.0,
    family_count_weight: float = 1.0,
    family_size_bins: Sequence[int] = DEFAULT_FAMILY_SIZE_BINS,
    refinement_passes: int = 4,
    swap_attempts_per_pass: int = 10000,
) -> Tuple[List[List[dict]], dict]:
    """Assign whole duplicate families while matching size, labels and family structure."""
    if min(size_weight, stratify_weight, family_structure_weight, family_count_weight) < 0:
        raise ValueError("assignment weights must be non-negative")

    rng = random.Random(seed)
    keys = base._infer_stratify_keys(data, stratify_keys)
    total = len(data)
    target_sizes = [float(r) * total for r in ratios]

    global_label_counts = Counter(base._stratum_token(item, keys) for item in data)
    label_strata = sorted(global_label_counts)
    target_label_counts = [
        {s: float(ratios[b]) * float(global_label_counts[s]) for s in label_strata}
        for b in range(3)
    ]

    family_sizes: List[int] = []
    family_label_counts: List[Counter] = []
    family_bin_names: List[str] = []
    global_bin_sample_counts = Counter()
    global_bin_family_counts = Counter()

    for members in families:
        size = len(members)
        family_sizes.append(size)
        family_label_counts.append(Counter(base._stratum_token(data[i], keys) for i in members))
        bin_name = _family_size_bin(size, family_size_bins)
        family_bin_names.append(bin_name)
        global_bin_sample_counts[bin_name] += size
        global_bin_family_counts[bin_name] += 1

    family_bins = sorted(global_bin_sample_counts)
    target_bin_sample_counts = [
        {b: float(ratios[p]) * float(global_bin_sample_counts[b]) for b in family_bins}
        for p in range(3)
    ]
    target_bin_family_counts = [
        {b: float(ratios[p]) * float(global_bin_family_counts[b]) for b in family_bins}
        for p in range(3)
    ]
    target_family_totals = [float(r) * len(families) for r in ratios]

    sizes = [0, 0, 0]
    label_counts = [Counter(), Counter(), Counter()]
    bin_sample_counts = [Counter(), Counter(), Counter()]
    bin_family_counts = [Counter(), Counter(), Counter()]
    assignment = [-1] * len(families)

    def objective() -> float:
        score = 0.0
        for p in range(3):
            target_size = max(1.0, target_sizes[p])
            score += size_weight * abs(float(sizes[p]) - target_sizes[p]) / target_size
            score += stratify_weight * _relative_l1(
                label_counts[p], target_label_counts[p], label_strata, target_size
            )
            score += family_structure_weight * _relative_l1(
                bin_sample_counts[p], target_bin_sample_counts[p], family_bins, target_size
            )
            target_family_total = max(1.0, target_family_totals[p])
            score += family_count_weight * _relative_l1(
                bin_family_counts[p], target_bin_family_counts[p], family_bins, target_family_total
            )
        return float(score)

    def apply_family(family_id: int, pool_id: int, sign: int) -> None:
        size = family_sizes[family_id]
        bin_name = family_bin_names[family_id]
        sizes[pool_id] += sign * size
        _counter_add(label_counts[pool_id], family_label_counts[family_id], sign)
        bin_sample_counts[pool_id][bin_name] += sign * size
        if bin_sample_counts[pool_id][bin_name] == 0:
            del bin_sample_counts[pool_id][bin_name]
        bin_family_counts[pool_id][bin_name] += sign
        if bin_family_counts[pool_id][bin_name] == 0:
            del bin_family_counts[pool_id][bin_name]

    order = list(range(len(families)))
    rng.shuffle(order)
    order.sort(key=lambda i: family_sizes[i], reverse=True)

    for family_id in order:
        candidates = [0, 1, 2]
        rng.shuffle(candidates)
        best_pool = candidates[0]
        best_score = float("inf")
        for pool_id in candidates:
            apply_family(family_id, pool_id, +1)
            score = objective()
            apply_family(family_id, pool_id, -1)
            if score < best_score - 1e-12:
                best_score = score
                best_pool = pool_id
        assignment[family_id] = best_pool
        apply_family(family_id, best_pool, +1)

    greedy_score = objective()
    accepted_moves = 0
    accepted_swaps = 0

    for _ in range(max(0, int(refinement_passes))):
        changed = False
        family_order = list(range(len(families)))
        rng.shuffle(family_order)

        for family_id in family_order:
            source = assignment[family_id]
            before = objective()
            apply_family(family_id, source, -1)
            best_pool = source
            best_score = before
            for destination in range(3):
                if destination == source:
                    continue
                apply_family(family_id, destination, +1)
                score = objective()
                apply_family(family_id, destination, -1)
                if score < best_score - 1e-12:
                    best_score = score
                    best_pool = destination
            apply_family(family_id, best_pool, +1)
            if best_pool != source:
                assignment[family_id] = best_pool
                accepted_moves += 1
                changed = True

        if len(families) >= 2:
            for _attempt in range(max(0, int(swap_attempts_per_pass))):
                left, right = rng.sample(range(len(families)), 2)
                lp = assignment[left]
                rp = assignment[right]
                if lp == rp:
                    continue
                before = objective()
                apply_family(left, lp, -1)
                apply_family(right, rp, -1)
                apply_family(left, rp, +1)
                apply_family(right, lp, +1)
                after = objective()
                if after < before - 1e-12:
                    assignment[left], assignment[right] = rp, lp
                    accepted_swaps += 1
                    changed = True
                else:
                    apply_family(left, rp, -1)
                    apply_family(right, lp, -1)
                    apply_family(left, lp, +1)
                    apply_family(right, rp, +1)

        if not changed:
            break

    buckets: List[List[dict]] = [[], [], []]
    pool_family_ids: List[List[int]] = [[], [], []]
    for family_id, members in enumerate(families):
        pool_id = assignment[family_id]
        pool_family_ids[pool_id].append(family_id)
        for idx in members:
            x = dict(data[idx])
            x["_strict_duplicate_group"] = int(family_id)
            x["_strict_family_size"] = int(len(members))
            x["_strict_family_size_bin"] = family_bin_names[family_id]
            buckets[pool_id].append(x)

    structure = _family_structure_report(families, assignment, ratios, family_size_bins)
    report = {
        "method": "group_aware_label_and_family_structure_stratification",
        "stratify_keys": list(keys),
        "size_weight": float(size_weight),
        "stratify_weight": float(stratify_weight),
        "family_structure_weight": float(family_structure_weight),
        "family_count_weight": float(family_count_weight),
        "family_size_bins": [int(x) for x in family_size_bins],
        "refinement_passes": int(refinement_passes),
        "swap_attempts_per_pass": int(swap_attempts_per_pass),
        "greedy_objective": float(greedy_score),
        "final_objective": float(objective()),
        "accepted_family_moves": int(accepted_moves),
        "accepted_family_swaps": int(accepted_swaps),
        "target_sizes": dict(zip(POOL_NAMES, target_sizes)),
        "actual_sizes": dict(zip(POOL_NAMES, [len(x) for x in buckets])),
        "family_counts": dict(zip(POOL_NAMES, [len(x) for x in pool_family_ids])),
        "family_structure": structure,
    }
    return buckets, report


def build_strict_pools(
    train_data: Sequence[dict],
    task_val: Sequence[dict],
    task_test: Sequence[dict],
    shadow_train_ratio: float = 0.40,
    shadow_val_ratio: float = 0.20,
    target_ratio: float = 0.40,
    seed: int = 42,
    group_key: Optional[str] = None,
    exact_deduplicate: bool = True,
    group_near_duplicates: bool = True,
    text_hamming_threshold: int = 3,
    image_hamming_threshold: int = 4,
    min_text_tokens_for_near: int = 3,
    stratify_keys: Optional[Sequence[str]] = None,
    size_weight: float = 1.0,
    stratify_weight: float = 2.0,
    family_structure_weight: float = 2.0,
    family_count_weight: float = 1.0,
    family_size_bins: Sequence[int] = DEFAULT_FAMILY_SIZE_BINS,
    refinement_passes: int = 4,
    swap_attempts_per_pass: int = 10000,
) -> Tuple[StrictPools, dict]:
    """Final strict protocol: exact dedup + family isolation + label/family stratification."""
    ratios = np.asarray([shadow_train_ratio, shadow_val_ratio, target_ratio], dtype=float)
    if np.any(ratios < 0) or not np.isclose(ratios.sum(), 1.0):
        raise ValueError("shadow_train_ratio + shadow_val_ratio + target_ratio must equal 1.")

    if exact_deduplicate:
        working, exact_report = base.remove_exact_duplicates(train_data)
    else:
        working, exact_report = base._annotate_without_exact_removal(train_data)

    keys = base._infer_stratify_keys(working, stratify_keys)
    families, duplicate_report = base.build_duplicate_families(
        working,
        group_key=group_key,
        group_near_duplicates=group_near_duplicates,
        text_hamming_threshold=text_hamming_threshold,
        image_hamming_threshold=image_hamming_threshold,
        min_text_tokens_for_near=min_text_tokens_for_near,
    )
    buckets, assignment_report = _assign_families_to_pools(
        working,
        families,
        ratios,
        seed,
        stratify_keys=keys,
        size_weight=size_weight,
        stratify_weight=stratify_weight,
        family_structure_weight=family_structure_weight,
        family_count_weight=family_count_weight,
        family_size_bins=family_size_bins,
        refinement_passes=refinement_passes,
        swap_attempts_per_pass=swap_attempts_per_pass,
    )

    pools = StrictPools(
        shadow_train=buckets[0],
        shadow_val=buckets[1],
        target=buckets[2],
        task_val=[dict(x) for x in task_val],
        task_test=[dict(x) for x in task_test],
    )

    strict_id_overlap = validate_pool_disjointness(pools)
    duplicate_group_overlap = base._validate_duplicate_groups_do_not_cross(pools)
    official_overlap = base._audit_official_exact_overlap(pools)
    label_distribution = base._label_distribution_report(working, pools, keys)

    report = {
        "seed": int(seed),
        "group_key": group_key,
        "ratios": dict(zip(POOL_NAMES, [float(x) for x in ratios])),
        "sizes": {
            "shadow_train": len(pools.shadow_train),
            "shadow_val": len(pools.shadow_val),
            "target": len(pools.target),
            "task_val": len(pools.task_val),
            "task_test": len(pools.task_test),
        },
        "exact_deduplication": exact_report,
        "near_duplicate_guard": duplicate_report,
        "assignment": assignment_report,
        "label_distribution": label_distribution,
        "family_structure_distribution": assignment_report["family_structure"],
        "strict_sample_id_overlap": strict_id_overlap,
        "duplicate_group_overlap": duplicate_group_overlap,
        "official_exact_overlap": official_overlap,
    }
    return pools, report
