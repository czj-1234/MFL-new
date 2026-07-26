import numpy as np
import torch

from src.acm_revision.data_protocol import make_client_specs, partition_clients_strict
from src.acm_revision.defenses import project_out
from src.acm_revision.strict_split import build_strict_pools, remove_exact_duplicates


def _binary_data(n=200, start=0, positive_every=2):
    return [
        {
            "id": start + i,
            "image": f"img/{start + i}.png",
            "text": f"example {start + i}",
            "label": 1 if i % positive_every == 0 else 0,
            "text_label": 1 if i % positive_every == 0 else 0,
            "image_label": 1 if i % positive_every == 0 else 0,
        }
        for i in range(n)
    ]


def test_strict_pools_are_disjoint_and_keep_unique_rows():
    data = _binary_data(200, start=0)
    pools, report = build_strict_pools(
        data,
        task_val=_binary_data(20, start=1000),
        task_test=_binary_data(20, start=2000),
        shadow_train_ratio=0.4,
        shadow_val_ratio=0.2,
        target_ratio=0.4,
        seed=42,
        group_near_duplicates=False,
        stratify_keys=["label"],
        refinement_passes=2,
        swap_attempts_per_pass=500,
    )
    a = {x["_strict_sample_id"] for x in pools.shadow_train}
    b = {x["_strict_sample_id"] for x in pools.shadow_val}
    c = {x["_strict_sample_id"] for x in pools.target}
    assert not (a & b)
    assert not (a & c)
    assert not (b & c)
    assert len(pools.shadow_train) + len(pools.shadow_val) + len(pools.target) == len(data)
    assert report["exact_deduplication"]["removed_examples"] == 0
    assert all(value == 0 for value in report["strict_sample_id_overlap"].values())
    assert all(value == 0 for value in report["duplicate_group_overlap"].values())


def test_exact_pair_duplicates_are_removed_but_text_only_repeats_are_kept():
    data = [
        {"id": "a", "image": "img/a.png", "text": "same words", "label": 0},
        {"id": "b", "image": "img/a.png", "text": "same words", "label": 0},
        {"id": "c", "image": "img/c.png", "text": "same words", "label": 1},
    ]
    retained, report = remove_exact_duplicates(data)
    assert [x["id"] for x in retained] == ["a", "c"]
    assert report["removed_examples"] == 1
    assert report["duplicate_exact_pair_rows_removed"] == 1


def test_repeated_text_is_grouped_not_deleted():
    data = _binary_data(40, start=0)
    data[0]["text"] = "same repeated meme template"
    data[1]["text"] = "same repeated meme template"
    data[0]["image"] = "img/a.png"
    data[1]["image"] = "img/b.png"

    pools, report = build_strict_pools(
        data,
        task_val=_binary_data(4, start=1000),
        task_test=_binary_data(4, start=2000),
        shadow_train_ratio=0.4,
        shadow_val_ratio=0.2,
        target_ratio=0.4,
        seed=42,
        group_near_duplicates=False,
        stratify_keys=["label"],
        refinement_passes=2,
        swap_attempts_per_pass=500,
    )

    all_rows = pools.shadow_train + pools.shadow_val + pools.target
    assert len(all_rows) == len(data)
    locations = {}
    for pool_name in ("shadow_train", "shadow_val", "target"):
        for item in getattr(pools, pool_name):
            if item["id"] in (0, 1):
                locations[item["id"]] = pool_name
    assert locations[0] == locations[1]
    assert report["near_duplicate_guard"]["exact_text_links"] >= 1
    assert report["exact_deduplication"]["removed_examples"] == 0


def test_group_aware_stratification_preserves_label_distribution():
    # 65/35 distribution, with a few repeated-text families that cannot be split.
    data = []
    for i in range(200):
        label = 0 if i < 130 else 1
        text = f"example {i}"
        if i in (0, 1, 2):
            text = "shared family zero"
        if i in (130, 131):
            text = "shared family one"
        data.append({
            "id": i,
            "image": f"img/{i}.png",
            "text": text,
            "label": label,
            "text_label": label,
            "image_label": label,
        })

    pools, report = build_strict_pools(
        data,
        task_val=_binary_data(10, start=1000),
        task_test=_binary_data(10, start=2000),
        seed=7,
        group_near_duplicates=False,
        stratify_keys=["label"],
        refinement_passes=3,
        swap_attempts_per_pass=1000,
    )
    assert len(pools.shadow_train) + len(pools.shadow_val) + len(pools.target) == 200
    assert report["label_distribution"]["max_absolute_joint_proportion_drift"] <= 0.02


def test_fixed_client_partition_has_no_reuse_and_exact_concentration():
    data = _binary_data(400)
    specs = make_client_specs(4, "modality_exclusive", 2)
    clients, manifest = partition_clients_strict(
        data,
        specs,
        num_classes=2,
        concentration=0.7,
        samples_per_client=50,
        seed=42,
        partition_mode="fixed",
    )
    ids = [x["_strict_sample_id"] for values in clients.values() for x in values]
    assert len(ids) == len(set(ids))
    assert manifest["allow_overlap"] is False
    for spec in specs:
        labels = [x["label"] for x in clients[spec.client_id]]
        assert labels.count(spec.dominant_label) == 35


def test_projection_removes_sensitive_direction():
    vector = np.asarray([3.0, 4.0], dtype=np.float32)
    basis = np.asarray([[1.0], [0.0]], dtype=np.float32)
    defended = project_out(vector, basis, alpha=1.0)
    assert np.allclose(defended, [0.0, 4.0], atol=1e-6)


def test_partial_projection_attenuates_not_deletes():
    vector = np.asarray([2.0, 1.0], dtype=np.float32)
    basis = np.asarray([[1.0], [0.0]], dtype=np.float32)
    defended = project_out(vector, basis, alpha=0.5)
    assert np.allclose(defended, [1.0, 1.0], atol=1e-6)
