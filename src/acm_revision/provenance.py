from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Optional

import torch


TRACKED_PACKAGES = [
    "torch",
    "torchvision",
    "transformers",
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "xgboost",
    "opacus",
    "Pillow",
    "PyYAML",
]


def _git_info() -> dict:
    result = {"commit": None, "branch": None, "dirty": None}
    try:
        result["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        result["branch"] = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], text=True)
        result["dirty"] = bool(status.strip())
    except Exception:
        pass
    return result


def capture_environment() -> dict:
    packages = {}
    for package in TRACKED_PACKAGES:
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None
    gpus = []
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            gpus.append(
                {
                    "index": idx,
                    "name": props.name,
                    "total_memory_bytes": int(props.total_memory),
                    "compute_capability": f"{props.major}.{props.minor}",
                }
            )
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpus": gpus,
        "packages": packages,
        "git": _git_info(),
    }


def save_environment(path: str | Path, extra: Optional[dict] = None) -> dict:
    payload = capture_environment()
    if extra:
        payload["experiment"] = extra
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload
