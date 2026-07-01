from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import random

from src.federated import (
    convert_to_client_sample,
    get_client_dominant_label,
    get_client_modality,
    get_sample_label_for_modality,
)


def _sample_pool(items, n, rng: random.Random):
    if not items:
        raise ValueError("Cannot sample from an empty bucket.")
    if len(items) >= n:
        return rng.sample(items, n)
    return [rng.choice(items) for _ in range(n)]


def build_strict_matched_pair(
    train_data,
    setting_name: str,
    num_clients: int,
    num_classes: int,
    samples_per_client: int,
    association: float = 0.7,
    split_seed: int = 4201,
):
    """Build associated and counterfactual client datasets from the same pools.

    For every modality pair, the same modality-specific sample pool is used in
    both conditions. The associated condition allocates samples according to the
    requested dominant-label ratio. The counterfactual condition redistributes
    the *same pool* evenly across the two same-modality clients, preserving:

    - client count and client size;
    - modality assignment;
    - modality-wise label marginals;
    - the union of sampled examples for each modality;
    - the global training seed and optimization configuration.

    This implementation targets the current 4-client / 2-class design:
    clients 0-1 are image clients and clients 2-3 are text clients.
    """
    if setting_name != "modality_exclusive":
        raise ValueError("Strict matched pairing currently supports modality_exclusive only.")
    if num_clients != 4 or num_classes != 2:
        raise ValueError("Strict matched pairing currently requires 4 clients and 2 classes.")
    if not 0.5 <= float(association) <= 1.0:
        raise ValueError("association must be in [0.5, 1.0].")

    rng = random.Random(split_seed)
    associated: Dict[int, List[dict]] = {i: [] for i in range(num_clients)}
    counterfactual: Dict[int, List[dict]] = {i: [] for i in range(num_clients)}

    modality_groups = {
        "image": [0, 1],
        "text": [2, 3],
    }

    for modality, client_ids in modality_groups.items():
        buckets = defaultdict(list)
        for item in train_data:
            label = get_sample_label_for_modality(item, modality)
            buckets[int(label)].append(item)

        # The two clients together need exactly 2 * samples_per_client samples.
        # To keep label marginals identical between associated and CF conditions,
        # sample one full client-size pool from each label.
        label0_pool = _sample_pool(buckets[0], samples_per_client, rng)
        label1_pool = _sample_pool(buckets[1], samples_per_client, rng)
        rng.shuffle(label0_pool)
        rng.shuffle(label1_pool)

        dominant_count = int(round(samples_per_client * float(association)))
        minority_count = samples_per_client - dominant_count

        # Associated allocation: client with dominant label 0 receives
        # dominant_count of label 0 and minority_count of label 1; vice versa.
        c0, c1 = client_ids
        assoc_raw = {
            c0: label0_pool[:dominant_count] + label1_pool[:minority_count],
            c1: label1_pool[minority_count:] + label0_pool[dominant_count:],
        }

        # Counterfactual allocation: exact same union pool, but balanced per client.
        half = samples_per_client // 2
        remainder = samples_per_client - half
        cf_raw = {
            c0: label0_pool[:half] + label1_pool[:remainder],
            c1: label0_pool[half:] + label1_pool[remainder:],
        }

        for cid in client_ids:
            rng.shuffle(assoc_raw[cid])
            rng.shuffle(cf_raw[cid])
            associated[cid] = [convert_to_client_sample(x, modality) for x in assoc_raw[cid]]
            counterfactual[cid] = [convert_to_client_sample(x, modality) for x in cf_raw[cid]]

    metadata = {
        "split_seed": split_seed,
        "association": float(association),
        "samples_per_client": samples_per_client,
        "associated_label_dist": {
            cid: dict(Counter(x["label"] for x in samples))
            for cid, samples in associated.items()
        },
        "counterfactual_label_dist": {
            cid: dict(Counter(x["label"] for x in samples))
            for cid, samples in counterfactual.items()
        },
    }
    return associated, counterfactual, metadata
