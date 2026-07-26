import numpy as np
import torch

from src.acm_revision.data_protocol import (
    ClientSpec,
    build_strict_pools,
    make_client_specs,
    partition_clients_strict,
)
from src.acm_revision.defenses import project_out


def _binary_data(n=200):
    return [
        {
            "id": i,
            "image": f"img/{i}.png",
            "text": f"example {i}",
            "label": i % 2,
            "text_label": i % 2,
            "image_label": i % 2,
        }
        for i in range(n)
    ]


def test_strict_pools_are_disjoint():
    data = _binary_data(200)
    pools, report = build_strict_pools(
        data,
        task_val=_binary_data(20),
        task_test=_binary_data(20),
        shadow_train_ratio=0.4,
        shadow_val_ratio=0.2,
        target_ratio=0.4,
        seed=42,
        deduplicate=False,
    )
    a = {x["_strict_sample_id"] for x in pools.shadow_train}
    b = {x["_strict_sample_id"] for x in pools.shadow_val}
    c = {x["_strict_sample_id"] for x in pools.target}
    assert not (a & b)
    assert not (a & c)
    assert not (b & c)
    assert all(value == 0 for value in report["overlap"].values())


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
