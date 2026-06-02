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