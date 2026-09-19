from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ClientSpec:
    client_id: int
    modality: str
    dominant_label: int


@dataclass
class StrictPools:
    shadow_train: List[dict]
    shadow_val: List[dict]
    target: List[dict]
    task_val: List[dict]
    task_test: List[dict]


def load_json(path: str | os.PathLike) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list at {path}, got {type(data).__name__}.")
    return data


def save_json(data, path: str | os.PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _normalise_text(text: str) -> str:
    text = str(text or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def _sample_text(item: Mapping) -> str:
    for key in ("text", "sentence", "caption", "content"):
        if item.get(key) is not None:
            return str(item.get(key))
    return ""


def _sample_image_path(item: Mapping) -> Optional[str]:
    for key in ("image", "image_path", "img", "path", "image_file", "filename"):
        value = item.get(key)
        if value:
            return str(value)
    return None


def sample_identity(item: Mapping, fallback_index: Optional[int] = None) -> str:
    """Stable paired-example identity used to prohibit cross-pool reuse.

    Prefer an explicit dataset ID. Otherwise hash the image path and normalized
    text together, which keeps the image-text pair as one indivisible example.
    """
    for key in ("id", "sample_id", "guid", "uid"):
        if key in item and item[key] not in (None, ""):
            return f"id:{item[key]}"
    image_path = _sample_image_path(item) or ""
    text = _normalise_text(_sample_text(item))
    payload = f"{image_path}|{text}|{fallback_index if fallback_index is not None else ''}"
    return "hash:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _simhash64(text: str) -> int:
    tokens = re.findall(r"\w+", _normalise_text(text))
    if not tokens:
        return 0
    acc = [0] * 64
    for token in tokens:
        h = int(hashlib.blake2b(token.encode("utf-8"), digest_size=8).hexdigest(), 16)
        for bit in range(64):
            acc[bit] += 1 if ((h >> bit) & 1) else -1
    out = 0
    for bit, score in enumerate(acc):
        if score >= 0:
            out |= 1 << bit
    return out


def _dhash64(image_path: Optional[str]) -> Optional[int]:
    if not image_path:
        return None
    path = Path(str(image_path))
    if not path.exists():
        candidates = [
            Path.cwd() / image_path,
            Path.cwd() / "data" / image_path,
            Path.cwd() / "data" / "raw" / image_path,
            Path.cwd() / "data" / "processed" / image_path,
            Path.cwd() / "data" / "images" / image_path,
            Path.cwd() / "data" / "MVSA" / image_path,
            Path.cwd() / "data" / "hateful_memes" / image_path,
            Path.cwd() / "data" / "Hateful Meme" / "hateful_memes" / image_path,
        ]
        path = next((p for p in candidates if p.exists()), path)
    try:
        with Image.open(path) as im:
            im = im.convert("L").resize((9, 8))
            a = np.asarray(im, dtype=np.int16)
        bits = a[:, 1:] > a[:, :-1]
        value = 0
        for idx, bit in enumerate(bits.reshape(-1).tolist()):
            if bit:
                value |= 1 << idx
        return value
    except Exception:
        return None


def _hamming64(a: Optional[int], b: Optional[int]) -> int:
    if a is None or b is None:
        return 65
    return int((a ^ b).bit_count())


def deduplicate_examples(
    data: Sequence[dict],
    text_hamming_threshold: int = 3,
    image_hamming_threshold: int = 4,
    enable_near_duplicate_check: bool = True,
) -> Tuple[List[dict], dict]:
    """Remove exact and conservative near duplicates.

    Near-duplicate candidates are only compared inside compact hash buckets to
    avoid an O(N^2) scan. A candidate is removed when either text SimHash or
    image dHash is extremely close. The returned report is persisted in split
    manifests so the paper can document overlap checks.
    """
    unique: List[dict] = []
    seen_ids = set()
    exact_removed = 0
    near_removed = 0
    text_buckets: MutableMapping[int, List[Tuple[int, int]]] = {}
    image_buckets: MutableMapping[int, List[Tuple[int, int]]] = {}

    for idx, item in enumerate(data):
        sid = sample_identity(item, idx)
        if sid in seen_ids:
            exact_removed += 1
            continue

        text_hash = _simhash64(_sample_text(item))
        image_hash = _dhash64(_sample_image_path(item))
        is_near = False

        if enable_near_duplicate_check:
            text_bucket = (text_hash >> 48) & 0xFFFF
            for _, other_hash in text_buckets.get(text_bucket, []):
                if _hamming64(text_hash, other_hash) <= text_hamming_threshold:
                    is_near = True
                    break

            if not is_near and image_hash is not None:
                image_bucket = (image_hash >> 48) & 0xFFFF
                for _, other_hash in image_buckets.get(image_bucket, []):
                    if _hamming64(image_hash, other_hash) <= image_hamming_threshold:
                        is_near = True
                        break

        if is_near:
            near_removed += 1
            continue

        new_item = dict(item)
        new_item["_strict_sample_id"] = sid
        unique.append(new_item)
        seen_ids.add(sid)

        text_bucket = (text_hash >> 48) & 0xFFFF
        text_buckets.setdefault(text_bucket, []).append((len(unique) - 1, text_hash))
        if image_hash is not None:
            image_bucket = (image_hash >> 48) & 0xFFFF
            image_buckets.setdefault(image_bucket, []).append((len(unique) - 1, image_hash))

    report = {
        "input_examples": len(data),
        "retained_examples": len(unique),
        "exact_duplicates_removed": exact_removed,
        "near_duplicates_removed": near_removed,
        "text_hamming_threshold": text_hamming_threshold,
        "image_hamming_threshold": image_hamming_threshold,
        "near_duplicate_check": enable_near_duplicate_check,
    }
    return unique, report


def _group_key(item: Mapping, key: Optional[str]) -> str:
    if key and item.get(key) not in (None, ""):
        return str(item[key])
    return str(item.get("_strict_sample_id") or sample_identity(item))


def build_strict_pools(
    train_data: Sequence[dict],
    task_val: Sequence[dict],
    task_test: Sequence[dict],
    shadow_train_ratio: float = 0.40,
    shadow_val_ratio: float = 0.20,
    target_ratio: float = 0.40,
    seed: int = 42,
    group_key: Optional[str] = None,
    deduplicate: bool = True,
    enable_near_duplicate_check: bool = True,
) -> Tuple[StrictPools, dict]:
    """Create completely disjoint shadow-train, shadow-val, and target pools.

    The official validation and test sets are never used to construct clients.
    Optional group-wise splitting can isolate source/topic/author/collection IDs.
    """
    ratios = np.asarray([shadow_train_ratio, shadow_val_ratio, target_ratio], dtype=float)
    if np.any(ratios < 0) or not np.isclose(ratios.sum(), 1.0):
        raise ValueError("shadow_train_ratio + shadow_val_ratio + target_ratio must equal 1.")

    working = [dict(x) for x in train_data]
    dedup_report = {"input_examples": len(working), "retained_examples": len(working)}
    if deduplicate:
        working, dedup_report = deduplicate_examples(
            working,
            enable_near_duplicate_check=enable_near_duplicate_check,
        )
    else:
        for idx, item in enumerate(working):
            item.setdefault("_strict_sample_id", sample_identity(item, idx))

    grouped: Dict[str, List[dict]] = {}
    for item in working:
        grouped.setdefault(_group_key(item, group_key), []).append(item)

    group_ids = list(grouped)
    rng = random.Random(seed)
    rng.shuffle(group_ids)

    total = sum(len(grouped[g]) for g in group_ids)
    targets = [ratios[0] * total, ratios[1] * total, ratios[2] * total]
    buckets: List[List[dict]] = [[], [], []]

    for gid in group_ids:
        group_items = grouped[gid]
        deficits = [targets[i] - len(buckets[i]) for i in range(3)]
        bucket_id = int(np.argmax(deficits))
        buckets[bucket_id].extend(group_items)

    pools = StrictPools(
        shadow_train=buckets[0],
        shadow_val=buckets[1],
        target=buckets[2],
        task_val=[dict(x) for x in task_val],
        task_test=[dict(x) for x in task_test],
    )
    overlap_report = validate_pool_disjointness(pools)
    report = {
        "seed": seed,
        "group_key": group_key,
        "ratios": {
            "shadow_train": shadow_train_ratio,
            "shadow_val": shadow_val_ratio,
            "target": target_ratio,
        },
        "sizes": {
            "shadow_train": len(pools.shadow_train),
            "shadow_val": len(pools.shadow_val),
            "target": len(pools.target),
            "task_val": len(pools.task_val),
            "task_test": len(pools.task_test),
        },
        "deduplication": dedup_report,
        "overlap": overlap_report,
    }
    return pools, report


def validate_pool_disjointness(pools: StrictPools) -> dict:
    client_pool_names = ("shadow_train", "shadow_val", "target")
    ids: Dict[str, set] = {}
    for name in client_pool_names:
        values = getattr(pools, name)
        ids[name] = {
            str(item.get("_strict_sample_id") or sample_identity(item, i))
            for i, item in enumerate(values)
        }

    overlaps = {}
    for i, left in enumerate(client_pool_names):
        for right in client_pool_names[i + 1 :]:
            overlap = ids[left].intersection(ids[right])
            overlaps[f"{left}__{right}"] = len(overlap)
            if overlap:
                raise ValueError(f"Strict pool overlap detected: {left} vs {right}: {len(overlap)} examples")
    return overlaps


def save_strict_pools(pools: StrictPools, report: dict, output_dir: str | os.PathLike) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(pools.shadow_train, output_dir / "shadow_train.json")
    save_json(pools.shadow_val, output_dir / "shadow_val.json")
    save_json(pools.target, output_dir / "target.json")
    save_json(pools.task_val, output_dir / "task_val.json")
    save_json(pools.task_test, output_dir / "task_test.json")
    save_json(report, output_dir / "split_report.json")


def make_client_specs(num_clients: int, setting_name: str, num_classes: int) -> List[ClientSpec]:
    if num_clients <= 0:
        raise ValueError("num_clients must be positive.")
    if num_classes <= 1:
        raise ValueError("num_classes must be >= 2.")

    setting_name = str(setting_name)
    specs: List[ClientSpec] = []
    for client_id in range(num_clients):
        if setting_name == "image_only":
            modality = "image"
            dominant = client_id % num_classes
        elif setting_name == "text_only":
            modality = "text"
            dominant = client_id % num_classes
        elif setting_name in ("full_multimodal", "complete_modality"):
            modality = "both"
            dominant = client_id % num_classes
        elif setting_name == "modality_exclusive":
            # Pair image/text clients for each dominant label, then repeat.
            modality = "image" if client_id % 2 == 0 else "text"
            dominant = (client_id // 2) % num_classes
        else:
            raise ValueError(f"Unknown setting_name: {setting_name}")
        specs.append(ClientSpec(client_id, modality, dominant))
    return specs


def sample_label(item: Mapping, modality: str) -> int:
    if modality == "text":
        if "text_label" in item:
            return int(item["text_label"])
        return int(item["label"])
    if modality == "image":
        if "image_label" in item:
            return int(item["image_label"])
        return int(item["label"])
    if modality == "both":
        if "label" in item:
            return int(item["label"])
        if "text_label" in item:
            return int(item["text_label"])
        return int(item["image_label"])
    raise ValueError(f"Unknown modality: {modality}")


def convert_for_client(item: Mapping, spec: ClientSpec) -> dict:
    out = dict(item)
    out["label"] = sample_label(item, spec.modality)
    out["modality"] = spec.modality
    out["client_id"] = spec.client_id
    out["dominant_label"] = spec.dominant_label
    return out


def _requested_label_counts(samples_per_client: int, num_classes: int, dominant_label: int, concentration: float) -> Dict[int, int]:
    if not 0.0 < concentration <= 1.0:
        raise ValueError("concentration must be in (0, 1].")
    dominant_count = int(round(samples_per_client * concentration))
    remaining = samples_per_client - dominant_count
    counts = {label: 0 for label in range(num_classes)}
    counts[dominant_label] = dominant_count
    other = [label for label in range(num_classes) if label != dominant_label]
    if other:
        base, extra = divmod(remaining, len(other))
        for label in other:
            counts[label] = base
        for label in other[:extra]:
            counts[label] += 1
    return counts


def partition_clients_strict(
    pool: Sequence[dict],
    specs: Sequence[ClientSpec],
    num_classes: int,
    concentration: float,
    samples_per_client: Optional[int],
    seed: int,
    partition_mode: str = "fixed",
) -> Tuple[Dict[int, List[dict]], dict]:
    """Partition a pool with zero raw-example reuse across clients.

    fixed: exact requested per-client concentration when feasible.
    full: assign every sample once, with probability weighted toward clients whose
          dominant class matches the sample under that client's modality.
    """
    rng = random.Random(seed)
    pool_items = [dict(x) for x in pool]
    for idx, item in enumerate(pool_items):
        item.setdefault("_strict_sample_id", sample_identity(item, idx))

    if len({x["_strict_sample_id"] for x in pool_items}) != len(pool_items):
        raise ValueError("Pool contains duplicate _strict_sample_id values.")

    client_data: Dict[int, List[dict]] = {spec.client_id: [] for spec in specs}

    if partition_mode == "full":
        shuffled = list(pool_items)
        rng.shuffle(shuffled)
        for item in shuffled:
            weights = []
            for spec in specs:
                label = sample_label(item, spec.modality)
                if math.isclose(concentration, 1.0):
                    weight = 1.0 if label == spec.dominant_label else 0.0
                else:
                    weight = concentration if label == spec.dominant_label else (1.0 - concentration) / max(1, num_classes - 1)
                weights.append(max(weight, 1e-12))
            total = sum(weights)
            r = rng.random() * total
            cumulative = 0.0
            selected = specs[-1]
            for spec, weight in zip(specs, weights):
                cumulative += weight
                if r <= cumulative:
                    selected = spec
                    break
            client_data[selected.client_id].append(convert_for_client(item, selected))
    elif partition_mode == "fixed":
        if samples_per_client is None:
            samples_per_client = len(pool_items) // len(specs)
        used_ids = set()
        shuffled_indices = list(range(len(pool_items)))
        rng.shuffle(shuffled_indices)

        # Rotate client order so early clients do not always receive the easiest allocation.
        client_order = list(specs)
        rng.shuffle(client_order)
        for spec in client_order:
            counts = _requested_label_counts(samples_per_client, num_classes, spec.dominant_label, concentration)
            chosen: List[dict] = []
            for label, n_needed in counts.items():
                candidates = [
                    pool_items[i]
                    for i in shuffled_indices
                    if pool_items[i]["_strict_sample_id"] not in used_ids
                    and sample_label(pool_items[i], spec.modality) == label
                ]
                if len(candidates) < n_needed:
                    raise ValueError(
                        f"Not enough disjoint samples for client={spec.client_id}, modality={spec.modality}, "
                        f"label={label}. Need {n_needed}, available {len(candidates)}. "
                        "Reduce samples_per_client or use a larger strict pool."
                    )
                selected = candidates[:n_needed]
                chosen.extend(selected)
                used_ids.update(x["_strict_sample_id"] for x in selected)
            rng.shuffle(chosen)
            client_data[spec.client_id] = [convert_for_client(x, spec) for x in chosen]
    else:
        raise ValueError("partition_mode must be 'fixed' or 'full'.")

    all_assigned_ids = [x["_strict_sample_id"] for values in client_data.values() for x in values]
    if len(all_assigned_ids) != len(set(all_assigned_ids)):
        raise AssertionError("Raw sample reuse across clients was detected.")

    manifest = {
        "seed": seed,
        "partition_mode": partition_mode,
        "concentration": concentration,
        "num_classes": num_classes,
        "samples_per_client": samples_per_client,
        "allow_overlap": False,
        "clients": [],
    }
    for spec in sorted(specs, key=lambda x: x.client_id):
        values = client_data[spec.client_id]
        counts = {str(label): 0 for label in range(num_classes)}
        for item in values:
            counts[str(int(item["label"]))] += 1
        manifest["clients"].append({
            **asdict(spec),
            "num_samples": len(values),
            "label_counts": counts,
            "sample_ids": [x["_strict_sample_id"] for x in values],
        })
    manifest["total_assigned"] = len(all_assigned_ids)
    manifest["unique_assigned"] = len(set(all_assigned_ids))
    return client_data, manifest


def create_matched_composition_pair(
    pool: Sequence[dict],
    spec: ClientSpec,
    num_classes: int,
    samples_per_client: int,
    concentrated: float,
    balanced: Optional[float] = None,
    seed: int = 42,
) -> Tuple[List[dict], List[dict], dict]:
    """Create concentrated/balanced local datasets for identical-checkpoint probes.

    Both branches draw from the same candidate pool and use a deterministic
    ordering. The overlap is maximized subject to their required label counts.
    """
    if balanced is None:
        balanced = 1.0 / num_classes
    rng = random.Random(seed)
    by_label: Dict[int, List[dict]] = {label: [] for label in range(num_classes)}
    for idx, item in enumerate(pool):
        x = dict(item)
        x.setdefault("_strict_sample_id", sample_identity(x, idx))
        label = sample_label(x, spec.modality)
        by_label[label].append(x)
    for label in by_label:
        rng.shuffle(by_label[label])

    conc_counts = _requested_label_counts(samples_per_client, num_classes, spec.dominant_label, concentrated)
    bal_counts = _requested_label_counts(samples_per_client, num_classes, spec.dominant_label, balanced)

    conc_raw: List[dict] = []
    bal_raw: List[dict] = []
    overlap_ids = set()
    for label in range(num_classes):
        need = max(conc_counts[label], bal_counts[label])
        if len(by_label[label]) < need:
            raise ValueError(
                f"Not enough label={label} examples for matched contrast. Need {need}, have {len(by_label[label])}."
            )
        shared_n = min(conc_counts[label], bal_counts[label])
        shared = by_label[label][:shared_n]
        rest = by_label[label][shared_n:need]
        conc_extra = rest[: conc_counts[label] - shared_n]
        bal_extra = rest[conc_counts[label] - shared_n : conc_counts[label] - shared_n + bal_counts[label] - shared_n]
        conc_raw.extend(shared + conc_extra)
        bal_raw.extend(shared + bal_extra)
        overlap_ids.update(x["_strict_sample_id"] for x in shared)

    rng.shuffle(conc_raw)
    rng.shuffle(bal_raw)
    conc = [convert_for_client(x, spec) for x in conc_raw]
    bal = [convert_for_client(x, spec) for x in bal_raw]
    report = {
        "client_id": spec.client_id,
        "modality": spec.modality,
        "dominant_label": spec.dominant_label,
        "concentrated": concentrated,
        "balanced": balanced,
        "samples_per_branch": samples_per_client,
        "shared_examples": len(overlap_ids),
        "shared_fraction": len(overlap_ids) / max(1, samples_per_client),
        "concentrated_counts": conc_counts,
        "balanced_counts": bal_counts,
    }
    return conc, bal, report
