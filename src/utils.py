# ============================================================
# Basic Utilities
# ============================================================

import os
import json
import random
from collections import Counter

import numpy as np
import torch


def set_seed(seed=42):
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json(path):
    """
    Load JSON file.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    """
    Save data to JSON file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def label_distribution(data):
    """
    Count label_name distribution.
    """
    return Counter([x["label_name"] for x in data])


def ensure_dir(path):
    """
    Create directory if it does not exist.
    """
    os.makedirs(path, exist_ok=True)


def load_config(path):
    """
    Load YAML config file.
    """
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================
# Class Imbalance Utilities
# ============================================================

def compute_class_weights_from_dataset(dataset, num_classes, device):
    """
    Compute global class weights for imbalanced classification.

    Formula:
        weight_c = total_samples / (num_classes * count_c)

    In federated learning, this should be computed from the global
    training dataset and shared across all clients.

    Args:
        dataset: training dataset
        num_classes: number of classes
        device: torch device

    Returns:
        class_weights: torch.FloatTensor, shape [num_classes]
    """

    labels = []

    for item in dataset:
        label = None

        # Case 1: dataset returns a dictionary
        if isinstance(item, dict):
            if "label" in item:
                label = item["label"]
            elif "labels" in item:
                label = item["labels"]
            elif "target" in item:
                label = item["target"]
            else:
                raise KeyError(
                    "Cannot find label key in dataset item. "
                    "Expected one of: label, labels, target."
                )

        # Case 2: dataset returns tuple/list
        elif isinstance(item, (tuple, list)):
            # Usually the label is the last element
            label = item[-1]

        else:
            raise TypeError(f"Unsupported dataset item type: {type(item)}")

        if isinstance(label, torch.Tensor):
            label = label.item()

        labels.append(int(label))

    label_counts = Counter(labels)
    total_samples = len(labels)

    weights = []

    for c in range(num_classes):
        count = label_counts.get(c, 0)

        if count == 0:
            # Avoid division by zero.
            # This should not happen for the global training dataset.
            weights.append(0.0)
        else:
            weights.append(total_samples / (num_classes * count))

    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    print("\n================ Class Imbalance Handling ================")
    print("Global class counts:", dict(label_counts))
    print("Global class weights:", class_weights.detach().cpu().tolist())
    print("==========================================================\n")

    return class_weights