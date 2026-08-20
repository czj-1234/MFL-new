from __future__ import annotations

import argparse
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict

from . import fl
from .data_protocol import (
    _requested_label_counts,
    convert_for_client,
    sample_identity,
    sample_label,
)
from .orchestrator import load_yaml


def _normalise_client_counts(raw, specs) -> dict[int, int]:
    """Resolve per-client sample counts from a mapping or a sequence."""
    if isinstance(raw, Mapping):
        out = {}
        for spec in specs:
            if spec.client_id in raw:
                value = raw[spec.client_id]
            elif str(spec.client_id) in raw:
                value = raw[str(spec.client_id)]
            else:
                raise ValueError(f"Missing client_sample_counts entry for client {spec.client_id}")
            out[spec.client_id] = int(value)
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        if len(raw) != len(specs):
            raise ValueError(
                f"client_sample_counts has {len(raw)} entries but {len(specs)} clients were requested"
            )
        out = {spec.client_id: int(value) for spec, value in zip(specs, raw)}
    else:
        raise TypeError("client_sample_counts must be a mapping or sequence")

    if any(value <= 0 for value in out.values()):
        raise ValueError("All client_sample_counts values must be positive")
    return out


def partition_clients_strict_variable(
    pool,
    specs,
    num_classes,
    concentration,
    samples_per_client,
    seed,
    partition_mode="fixed",
    *,
    client_sample_counts,
):
    """Strict fixed partition with variable client sizes and zero sample reuse.

    This diagnostic-only wrapper preserves the normal strict partition semantics,
    but lets us keep the system-wide number of assigned examples exactly fixed
    when the number of clients does not divide that total evenly.
    """
    if partition_mode != "fixed":
        raise ValueError("Variable client sizes are supported only with partition_mode='fixed'.")

    rng = random.Random(seed)
    pool_items = [dict(x) for x in pool]
    for idx, item in enumerate(pool_items):
        item.setdefault("_strict_sample_id", sample_identity(item, idx))

    if len({x["_strict_sample_id"] for x in pool_items}) != len(pool_items):
        raise ValueError("Pool contains duplicate _strict_sample_id values.")

    per_client = _normalise_client_counts(client_sample_counts, specs)
    client_data = {spec.client_id: [] for spec in specs}
    used_ids = set()
    shuffled_indices = list(range(len(pool_items)))
    rng.shuffle(shuffled_indices)

    client_order = list(specs)
    rng.shuffle(client_order)
    for spec in client_order:
        n_samples = per_client[spec.client_id]
        counts = _requested_label_counts(n_samples, num_classes, spec.dominant_label, concentration)
        chosen = []
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
                    f"label={label}. Need {n_needed}, available {len(candidates)}."
                )
            selected = candidates[:n_needed]
            chosen.extend(selected)
            used_ids.update(x["_strict_sample_id"] for x in selected)
        rng.shuffle(chosen)
        client_data[spec.client_id] = [convert_for_client(x, spec) for x in chosen]

    all_assigned_ids = [x["_strict_sample_id"] for values in client_data.values() for x in values]
    if len(all_assigned_ids) != len(set(all_assigned_ids)):
        raise AssertionError("Raw sample reuse across clients was detected.")

    manifest = {
        "seed": seed,
        "partition_mode": partition_mode,
        "concentration": concentration,
        "num_classes": num_classes,
        "samples_per_client": None,
        "client_sample_counts": {str(k): v for k, v in sorted(per_client.items())},
        "allow_overlap": False,
        "clients": [],
    }
    for spec in sorted(specs, key=lambda x: x.client_id):
        values = client_data[spec.client_id]
        counts = {str(label): 0 for label in range(num_classes)}
        for item in values:
            counts[str(int(item["label"]))] += 1
        manifest["clients"].append(
            {
                **asdict(spec),
                "num_samples": len(values),
                "label_counts": counts,
                "sample_ids": [x["_strict_sample_id"] for x in values],
            }
        )
    manifest["total_assigned"] = len(all_assigned_ids)
    manifest["unique_assigned"] = len(set(all_assigned_ids))
    return client_data, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--population", default=None, choices=["shadow_train", "shadow_val", "target"])
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    raw_counts = cfg.get("federated", {}).get("client_sample_counts")
    if raw_counts is None:
        raise SystemExit("Config must define federated.client_sample_counts")

    original_partition = fl.partition_clients_strict

    def patched_partition(pool, specs, num_classes, concentration, samples_per_client, seed, partition_mode="fixed"):
        return partition_clients_strict_variable(
            pool,
            specs,
            num_classes,
            concentration,
            samples_per_client,
            seed,
            partition_mode,
            client_sample_counts=raw_counts,
        )

    fl.partition_clients_strict = patched_partition
    try:
        population = args.population or cfg.get("experiment", {}).get("population", "target")
        result = fl.run_strict_fl(cfg, population=population)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        fl.partition_clients_strict = original_partition


if __name__ == "__main__":
    main()
