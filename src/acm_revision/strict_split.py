from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from functools import lru_cache
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


POOL_NAMES = ("shadow_train", "shadow_val", "target")


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


def _resolve_image_path(image_path: Optional[str]) -> Optional[Path]:
    if not image_path:
        return None
    raw = Path(str(image_path))
    candidates = [
        raw,
        Path.cwd() / raw,
        Path.cwd() / "data" / raw,
        Path.cwd() / "data" / "raw" / raw,
        Path.cwd() / "data" / "processed" / raw,
        Path.cwd() / "data" / "images" / raw,
        Path.cwd() / "data" / "MVSA" / raw,
        Path.cwd() / "data" / "hateful_memes" / raw,
        Path.cwd() / "data" / "Hateful Meme" / "hateful_memes" / raw,
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    return None


@lru_cache(maxsize=65536)
def _image_exact_signature(image_path: Optional[str]) -> str:
    """Exact image signature; fall back to the normalized path if bytes are unavailable."""
    if not image_path:
        return "image:none"
    resolved = _resolve_image_path(image_path)
    if resolved is not None:
        digest = hashlib.sha256()
        try:
            with resolved.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    digest.update(chunk)
            return "image:sha256:" + digest.hexdigest()
        except OSError:
            pass
    return "image:path:" + str(Path(str(image_path)).as_posix()).lower()


def _exact_pair_fingerprint(item: Mapping) -> str:
    text = _normalise_text(_sample_text(item))
    image_sig = _image_exact_signature(_sample_image_path(item))
    payload = json.dumps([image_sig, text], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _base_identity(item: Mapping) -> str:
    # Calling without a fallback index keeps the identity stable across files/pools.
    return str(sample_identity(item))


def remove_exact_duplicates(data: Sequence[dict]) -> Tuple[List[dict], dict]:
    """Remove true exact duplicates, never text-only or image-only repetitions.

    A row is removed when either its explicit/base dataset identity has already
    appeared or the exact paired image+text fingerprint has already appeared.
    Repeated text with a different image and repeated images with different text
    are retained and handled later as split-level duplicate families.
    """
    retained: List[dict] = []
    seen_base: Dict[str, int] = {}
    seen_pair: Dict[str, int] = {}
    duplicate_base_rows = 0
    duplicate_pair_rows = 0

    for item in data:
        x = dict(item)
        base = _base_identity(x)
        pair = _exact_pair_fingerprint(x)

        if base in seen_base:
            duplicate_base_rows += 1
            continue
        if pair in seen_pair:
            duplicate_pair_rows += 1
            continue

        x["_strict_base_identity"] = base
        x["_strict_pair_fingerprint"] = pair
        x["_strict_sample_id"] = base
        seen_base[base] = len(retained)
        seen_pair[pair] = len(retained)
        retained.append(x)

    removed = len(data) - len(retained)
    return retained, {
        "input_examples": len(data),
        "retained_examples": len(retained),
        "removed_examples": removed,
        "duplicate_base_identity_rows_removed": duplicate_base_rows,
        "duplicate_exact_pair_rows_removed": duplicate_pair_rows,
        "policy": "remove_exact_pair_or_identity_only",
    }


def _annotate_without_exact_removal(data: Sequence[dict]) -> Tuple[List[dict], dict]:
    """Ablation path that keeps rows but still gives every row a unique strict ID."""
    out: List[dict] = []
    occurrences: Counter[str] = Counter()
    for item in data:
        x = dict(item)
        base = _base_identity(x)
        occurrence = occurrences[base]
        occurrences[base] += 1
        strict_id = base if occurrence == 0 else f"{base}::occurrence={occurrence}"
        x["_strict_base_identity"] = base
        x["_strict_pair_fingerprint"] = _exact_pair_fingerprint(x)
        x["_strict_sample_id"] = strict_id
        out.append(x)
    return out, {
        "input_examples": len(data),
        "retained_examples": len(out),
        "removed_examples": 0,
        "duplicate_base_identity_rows_retained": int(sum(max(0, n - 1) for n in occurrences.values())),
        "policy": "keep_exact_duplicates_ablation",
    }


def _token_count(text: str) -> int:
    return len([t for t in _normalise_text(text).split() if t])


def _band_keys(value: int, bands: int = 8) -> List[Tuple[int, int]]:
    """LSH bands used only to generate candidates before exact Hamming checking."""
    if 64 % bands != 0:
        raise ValueError("bands must divide 64")
    width = 64 // bands
    mask = (1 << width) - 1
    return [(band, (value >> (band * width)) & mask) for band in range(bands)]


def build_duplicate_families(
    data: Sequence[dict],
    *,
    group_key: Optional[str] = None,
    group_near_duplicates: bool = True,
    text_hamming_threshold: int = 3,
    image_hamming_threshold: int = 4,
    min_text_tokens_for_near: int = 3,
) -> Tuple[List[List[int]], dict]:
    """Build indivisible split families without deleting near-duplicate examples.

    Exact repeated text/image and conservative near-text/near-image matches are
    linked into one family. A family can appear in only one of Shadow-Train,
    Shadow-Val or Target. This blocks attack train/test leakage while retaining
    legitimate benchmark examples such as the same text paired with another image.
    """
    n = len(data)
    uf = _UnionFind(n)
    stats = Counter()

    first_by_base: Dict[str, int] = {}
    for idx, item in enumerate(data):
        base = str(item.get("_strict_base_identity") or _base_identity(item))
        if base in first_by_base:
            if uf.union(idx, first_by_base[base]):
                stats["base_identity_links"] += 1
        else:
            first_by_base[base] = idx

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

    # Repeated text templates are kept, but cannot straddle strict pools.
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

    # Repeated exact images are kept when the paired text differs, but stay in one pool.
    first_by_image: Dict[str, int] = {}
    for idx, item in enumerate(data):
        image_path = _sample_image_path(item)
        if not image_path:
            continue
        signature = _image_exact_signature(image_path)
        if signature in first_by_image:
            if uf.union(idx, first_by_image[signature]):
                stats["exact_image_links"] += 1
        else:
            first_by_image[signature] = idx

    if group_near_duplicates:
        text_index: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        text_hashes: Dict[int, Tuple[int, int]] = {}
        image_index: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        image_hashes: Dict[int, int] = {}

        for idx, item in enumerate(data):
            text = _sample_text(item)
            norm_text = _normalise_text(text)
            token_count = _token_count(text)
            if norm_text and token_count >= min_text_tokens_for_near:
                text_hash = _simhash64(text)
                candidates = set()
                for key in _band_keys(text_hash):
                    candidates.update(text_index.get(key, []))
                for other_idx in candidates:
                    other_hash, other_tokens = text_hashes[other_idx]
                    if min(token_count, other_tokens) < min_text_tokens_for_near:
                        continue
                    if _hamming64(text_hash, other_hash) <= text_hamming_threshold:
                        if uf.union(idx, other_idx):
                            stats["near_text_links"] += 1
                text_hashes[idx] = (text_hash, token_count)
                for key in _band_keys(text_hash):
                    text_index[key].append(idx)

            image_hash = _dhash64(_sample_image_path(item))
            if image_hash is not None:
                candidates = set()
                for key in _band_keys(image_hash):
                    candidates.update(image_index.get(key, []))
                for other_idx in candidates:
                    other_hash = image_hashes[other_idx]
                    if _hamming64(image_hash, other_hash) <= image_hamming_threshold:
                        if uf.union(idx, other_idx):
                            stats["near_image_links"] += 1
                image_hashes[idx] = image_hash
                for key in _band_keys(image_hash):
                    image_index[key].append(idx)

    grouped: Dict[int, List[int]] = defaultdict(list)
    for idx in range(n):
        grouped[uf.find(idx)].append(idx)

    families = list(grouped.values())
    sizes = [len(x) for x in families]
    multi = [s for s in sizes if s > 1]
    report = {
        "mode": "exact_remove_then_near_group",
        "group_near_duplicates": bool(group_near_duplicates),
        "text_hamming_threshold": int(text_hamming_threshold),
        "image_hamming_threshold": int(image_hamming_threshold),
        "min_text_tokens_for_near": int(min_text_tokens_for_near),
        "num_families": len(families),
        "num_multi_example_families": len(multi),
        "largest_family_size": max(sizes) if sizes else 0,
        "examples_in_multi_example_families": int(sum(multi)),
        **{k: int(v) for k, v in stats.items()},
    }
    return families, report


def _infer_stratify_keys(data: Sequence[Mapping], explicit: Optional[Sequence[str]]) -> List[str]:
    if explicit:
        keys = [str(k) for k in explicit if str(k)]
        missing = [k for k in keys if any(k not in item for item in data)]
        if missing:
            raise ValueError(f"stratify_keys missing from one or more examples: {missing}")
        return keys
    if data and all("label" in item for item in data):
        return ["label"]
    if data and all("image_label" in item and "text_label" in item for item in data):
        return ["image_label", "text_label"]
    return []


def _stratum_token(item: Mapping, keys: Sequence[str]) -> str:
    if not keys:
        return "__all__"
    return json.dumps([item.get(k) for k in keys], ensure_ascii=False, separators=(",", ":"))


def _counter_add(target: Counter, source: Counter, sign: int = 1) -> None:
    for key, value in source.items():
        target[key] += sign * int(value)
        if target[key] == 0:
            del target[key]


def _assignment_objective(
    sizes: Sequence[int],
    counts: Sequence[Counter],
    target_sizes: Sequence[float],
    target_counts: Sequence[Mapping[str, float]],
    strata: Sequence[str],
    *,
    size_weight: float,
    stratify_weight: float,
) -> float:
    score = 0.0
    for bucket_id in range(3):
        target_size = max(1.0, float(target_sizes[bucket_id]))
        score += size_weight * abs(float(sizes[bucket_id]) - target_sizes[bucket_id]) / target_size
        if strata:
            l1 = sum(
                abs(float(counts[bucket_id].get(s, 0)) - float(target_counts[bucket_id].get(s, 0.0)))
                for s in strata
            )
            score += stratify_weight * l1 / (2.0 * target_size)
    return float(score)


def _assign_families_to_pools(
    data: Sequence[dict],
    families: Sequence[Sequence[int]],
    ratios: Sequence[float],
    seed: int,
    *,
    stratify_keys: Optional[Sequence[str]] = None,
    size_weight: float = 1.0,
    stratify_weight: float = 2.0,
    refinement_passes: int = 3,
    swap_attempts_per_pass: int = 5000,
) -> Tuple[List[List[dict]], dict]:
    """Group-aware stratified assignment with whole-family refinement."""
    if size_weight < 0 or stratify_weight < 0:
        raise ValueError("size_weight and stratify_weight must be non-negative")

    rng = random.Random(seed)
    keys = _infer_stratify_keys(data, stratify_keys)
    total = len(data)
    target_sizes = [float(r) * total for r in ratios]

    global_counts = Counter(_stratum_token(item, keys) for item in data)
    strata = sorted(global_counts)
    target_counts = [
        {s: float(ratios[b]) * float(global_counts[s]) for s in strata}
        for b in range(3)
    ]

    family_sizes: List[int] = []
    family_counts: List[Counter] = []
    for members in families:
        family_sizes.append(len(members))
        family_counts.append(Counter(_stratum_token(data[i], keys) for i in members))

    order = list(range(len(families)))
    rng.shuffle(order)
    order.sort(key=lambda i: family_sizes[i], reverse=True)

    sizes = [0, 0, 0]
    counts = [Counter(), Counter(), Counter()]
    assignment = [-1] * len(families)

    def objective() -> float:
        return _assignment_objective(
            sizes,
            counts,
            target_sizes,
            target_counts,
            strata,
            size_weight=size_weight,
            stratify_weight=stratify_weight,
        )

    for family_id in order:
        size = family_sizes[family_id]
        strata_count = family_counts[family_id]
        pool_order = [0, 1, 2]
        rng.shuffle(pool_order)
        best_score = float("inf")
        best_pool = pool_order[0]
        for pool_id in pool_order:
            sizes[pool_id] += size
            _counter_add(counts[pool_id], strata_count, +1)
            score = objective()
            sizes[pool_id] -= size
            _counter_add(counts[pool_id], strata_count, -1)
            if score < best_score - 1e-12:
                best_score = score
                best_pool = pool_id
        assignment[family_id] = best_pool
        sizes[best_pool] += size
        _counter_add(counts[best_pool], strata_count, +1)

    greedy_score = objective()
    accepted_moves = 0
    accepted_swaps = 0

    for _ in range(max(0, int(refinement_passes))):
        changed = False
        family_order = list(range(len(families)))
        rng.shuffle(family_order)

        for family_id in family_order:
            source = assignment[family_id]
            size = family_sizes[family_id]
            strata_count = family_counts[family_id]
            before = objective()

            sizes[source] -= size
            _counter_add(counts[source], strata_count, -1)
            best_pool = source
            best_score = before
            for destination in range(3):
                if destination == source:
                    continue
                sizes[destination] += size
                _counter_add(counts[destination], strata_count, +1)
                score = objective()
                sizes[destination] -= size
                _counter_add(counts[destination], strata_count, -1)
                if score < best_score - 1e-12:
                    best_score = score
                    best_pool = destination

            sizes[best_pool] += size
            _counter_add(counts[best_pool], strata_count, +1)
            if best_pool != source:
                assignment[family_id] = best_pool
                accepted_moves += 1
                changed = True

        if len(families) >= 2:
            for _attempt in range(max(0, int(swap_attempts_per_pass))):
                left, right = rng.sample(range(len(families)), 2)
                left_pool = assignment[left]
                right_pool = assignment[right]
                if left_pool == right_pool:
                    continue

                before = objective()
                ls, rs = family_sizes[left], family_sizes[right]
                lc, rc = family_counts[left], family_counts[right]

                sizes[left_pool] -= ls
                _counter_add(counts[left_pool], lc, -1)
                sizes[right_pool] -= rs
                _counter_add(counts[right_pool], rc, -1)
                sizes[left_pool] += rs
                _counter_add(counts[left_pool], rc, +1)
                sizes[right_pool] += ls
                _counter_add(counts[right_pool], lc, +1)

                after = objective()
                if after < before - 1e-12:
                    assignment[left], assignment[right] = right_pool, left_pool
                    accepted_swaps += 1
                    changed = True
                else:
                    sizes[left_pool] -= rs
                    _counter_add(counts[left_pool], rc, -1)
                    sizes[right_pool] -= ls
                    _counter_add(counts[right_pool], lc, -1)
                    sizes[left_pool] += ls
                    _counter_add(counts[left_pool], lc, +1)
                    sizes[right_pool] += rs
                    _counter_add(counts[right_pool], rc, +1)

        if not changed:
            break

    buckets: List[List[dict]] = [[], [], []]
    bucket_family_ids: List[List[int]] = [[], [], []]
    for family_id, members in enumerate(families):
        pool_id = assignment[family_id]
        bucket_family_ids[pool_id].append(family_id)
        for idx in members:
            x = dict(data[idx])
            x["_strict_duplicate_group"] = int(family_id)
            buckets[pool_id].append(x)

    report = {
        "method": "group_aware_stratified_greedy_plus_family_refinement",
        "stratify_keys": keys,
        "size_weight": float(size_weight),
        "stratify_weight": float(stratify_weight),
        "refinement_passes": int(refinement_passes),
        "swap_attempts_per_pass": int(swap_attempts_per_pass),
        "greedy_objective": float(greedy_score),
        "final_objective": float(objective()),
        "accepted_family_moves": int(accepted_moves),
        "accepted_family_swaps": int(accepted_swaps),
        "target_sizes": dict(zip(POOL_NAMES, target_sizes)),
        "actual_sizes": dict(zip(POOL_NAMES, [len(x) for x in buckets])),
        "family_counts": dict(zip(POOL_NAMES, [len(x) for x in bucket_family_ids])),
    }
    return buckets, report


def _validate_duplicate_groups_do_not_cross(pools: StrictPools) -> dict:
    groups = {
        name: {int(x["_strict_duplicate_group"]) for x in getattr(pools, name)}
        for name in POOL_NAMES
    }
    report = {}
    for i, left in enumerate(POOL_NAMES):
        for right in POOL_NAMES[i + 1 :]:
            overlap = groups[left].intersection(groups[right])
            report[f"{left}__{right}"] = len(overlap)
            if overlap:
                raise ValueError(
                    f"Duplicate/source families crossed strict pools: {left} vs {right}: {len(overlap)} groups"
                )
    return report


def _identity_set(data: Sequence[Mapping]) -> set[str]:
    return {_base_identity(x) for x in data}


def _pair_set(data: Sequence[Mapping]) -> set[str]:
    return {_exact_pair_fingerprint(x) for x in data}


def _audit_official_exact_overlap(pools: StrictPools) -> dict:
    identity_sets = {
        "shadow_train": {str(x["_strict_base_identity"]) for x in pools.shadow_train},
        "shadow_val": {str(x["_strict_base_identity"]) for x in pools.shadow_val},
        "target": {str(x["_strict_base_identity"]) for x in pools.target},
        "task_val": _identity_set(pools.task_val),
        "task_test": _identity_set(pools.task_test),
    }
    pair_sets = {
        "shadow_train": {str(x["_strict_pair_fingerprint"]) for x in pools.shadow_train},
        "shadow_val": {str(x["_strict_pair_fingerprint"]) for x in pools.shadow_val},
        "target": {str(x["_strict_pair_fingerprint"]) for x in pools.target},
        "task_val": _pair_set(pools.task_val),
        "task_test": _pair_set(pools.task_test),
    }

    identity_report = {}
    pair_report = {}
    names = list(identity_sets)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            key = f"{left}__{right}"
            identity_report[key] = len(identity_sets[left].intersection(identity_sets[right]))
            pair_report[key] = len(pair_sets[left].intersection(pair_sets[right]))

    bad_identity = {k: v for k, v in identity_report.items() if v > 0}
    bad_pair = {k: v for k, v in pair_report.items() if v > 0}
    if bad_identity or bad_pair:
        raise ValueError(
            "Exact overlap across official/strict splits: "
            f"identity={bad_identity}, exact_pair={bad_pair}"
        )
    return {"identity": identity_report, "exact_pair": pair_report}


def _distribution(data: Sequence[Mapping], keys: Sequence[str]) -> dict:
    total = len(data)
    joint = Counter(_stratum_token(x, keys) for x in data)
    marginals = {}
    for key in keys:
        c = Counter(str(x.get(key)) for x in data)
        marginals[key] = {
            "counts": dict(sorted(c.items())),
            "proportions": {k: (v / total if total else 0.0) for k, v in sorted(c.items())},
        }
    return {
        "n": total,
        "joint_counts": dict(sorted(joint.items())),
        "joint_proportions": {k: (v / total if total else 0.0) for k, v in sorted(joint.items())},
        "marginals": marginals,
    }


def _label_distribution_report(reference: Sequence[dict], pools: StrictPools, keys: Sequence[str]) -> dict:
    ref = _distribution(reference, keys)
    ref_joint = ref["joint_proportions"]
    pool_reports = {}
    max_drift = 0.0
    for name in POOL_NAMES:
        dist = _distribution(getattr(pools, name), keys)
        all_strata = set(ref_joint) | set(dist["joint_proportions"])
        drift = max(
            (
                abs(float(dist["joint_proportions"].get(s, 0.0)) - float(ref_joint.get(s, 0.0)))
                for s in all_strata
            ),
            default=0.0,
        )
        dist["max_absolute_joint_proportion_drift"] = float(drift)
        max_drift = max(max_drift, drift)
        pool_reports[name] = dist
    return {
        "stratify_keys": list(keys),
        "reference_after_exact_dedup": ref,
        "pools": pool_reports,
        "max_absolute_joint_proportion_drift": float(max_drift),
    }


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
    refinement_passes: int = 3,
    swap_attempts_per_pass: int = 5000,
) -> Tuple[StrictPools, dict]:
    """Create reviewer-compliant strict Shadow/Target pools.

    1) Remove exact duplicate identities / exact image-text pairs only.
    2) Retain near-duplicate/repeated-template examples, but group them into an
       indivisible family that cannot cross Shadow-Train/Shadow-Val/Target.
    3) Assign whole families with group-aware stratification so label distributions
       stay close to the post-dedup benchmark distribution.
    4) Keep official validation/test data outside every client-construction pool.
    """
    ratios = np.asarray([shadow_train_ratio, shadow_val_ratio, target_ratio], dtype=float)
    if np.any(ratios < 0) or not np.isclose(ratios.sum(), 1.0):
        raise ValueError("shadow_train_ratio + shadow_val_ratio + target_ratio must equal 1.")

    if exact_deduplicate:
        working, exact_report = remove_exact_duplicates(train_data)
    else:
        working, exact_report = _annotate_without_exact_removal(train_data)

    keys = _infer_stratify_keys(working, stratify_keys)
    families, duplicate_report = build_duplicate_families(
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
    duplicate_group_overlap = _validate_duplicate_groups_do_not_cross(pools)
    official_overlap = _audit_official_exact_overlap(pools)
    distribution_report = _label_distribution_report(working, pools, keys)

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
        "label_distribution": distribution_report,
        "strict_sample_id_overlap": strict_id_overlap,
        "duplicate_group_overlap": duplicate_group_overlap,
        "official_exact_overlap": official_overlap,
    }
    return pools, report
