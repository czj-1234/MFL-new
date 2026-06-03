# ============================================================
# MVSA Dataset for 6-Class Multimodal Federated Learning
# Compatible with:
#   1. json_path input
#   2. Python list input
#   3. image / text / both modality modes
#   4. ResNet18 image tensor format
#   5. DistilBERT tokenizer output
#   6. old imports: MVSAVoteDataset, load_mvsa_datasets
# ============================================================

import json
from pathlib import Path

from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


# ============================================================
# Image Transform
# ============================================================

image_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# Helper
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# Dataset
# ============================================================

class MVSAStrongDataset(Dataset):
    """
    MVSA 6-class dataset.

    Supports:
        data=list
        json_path=str
        first positional argument as either list or json_path
    """

    def __init__(
        self,
        data=None,
        tokenizer=None,
        mode="both",
        max_text_len=64,
        image_transform=None,
        json_path=None,
        transform=None,
        max_len=None,
        max_length=None,
        **kwargs,
    ):
        super().__init__()

        # ------------------------------------------------------------
        # Backward compatibility:
        # If old code calls MVSAVoteDataset(json_path, tokenizer=...)
        # then data receives a string path. Convert it to json_path.
        # ------------------------------------------------------------
        if isinstance(data, (str, Path)):
            json_path = data
            data = None

        # ------------------------------------------------------------
        # Load data
        # ------------------------------------------------------------
        if data is None:
            if json_path is None:
                json_path = kwargs.get("path", None)

            if json_path is None:
                raise ValueError(
                    "MVSAStrongDataset requires either data=list or json_path=str."
                )

            data = load_json(json_path)

        self.data = data
        self.tokenizer = tokenizer

        if mode not in ["image", "text", "both"]:
            raise ValueError(f"Unknown mode: {mode}. Expected image, text, or both.")

        self.mode = mode

        # compatible names: max_text_len / max_len / max_length
        if max_len is not None:
            max_text_len = max_len
        if max_length is not None:
            max_text_len = max_length

        self.max_text_len = int(max_text_len)

        # compatible names: image_transform / transform
        if image_transform is None and transform is not None:
            image_transform = transform

        if image_transform is None:
            image_transform = globals()["image_transform"]

        self.image_transform = image_transform

    def __len__(self):
        return len(self.data)

    def _resolve_image_path(self, image_path):
        if image_path is None:
            return None

        image_path = str(image_path).strip()

        if image_path == "":
            return None

        p = Path(image_path)

        if p.exists():
            return p

        # Try relative to project root/current working directory
        p2 = Path.cwd() / image_path

        if p2.exists():
            return p2

        # Try common MVSA image folders
        candidates = [
            Path.cwd() / "data" / image_path,
            Path.cwd() / "data" / "raw" / image_path,
            Path.cwd() / "data" / "processed" / image_path,
            Path.cwd() / "data" / "images" / image_path,
        ]

        for c in candidates:
            if c.exists():
                return c

        return p

    def _load_image(self, image_path):
        """
        Load image and return tensor [3, 224, 224].
        If loading fails, return zero tensor.
        """
        try:
            resolved_path = self._resolve_image_path(image_path)

            if resolved_path is None:
                return torch.zeros(3, 224, 224)

            image = Image.open(resolved_path).convert("RGB")

            if self.image_transform is not None:
                image = self.image_transform(image)

            if isinstance(image, Image.Image):
                image = globals()["image_transform"](image)

            return image

        except Exception:
            return torch.zeros(3, 224, 224)

    def _encode_text(self, text):
        """
        Tokenize text and return input_ids and attention_mask.
        If tokenizer is missing, return zero tensors.
        """
        if self.tokenizer is None:
            input_ids = torch.zeros(self.max_text_len, dtype=torch.long)
            attention_mask = torch.zeros(self.max_text_len, dtype=torch.long)
            return input_ids, attention_mask

        encoded = self.tokenizer(
            str(text),
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        return input_ids, attention_mask

    def __getitem__(self, idx):
        item = self.data[idx]

        label = int(item["label"])

        image_path = (
            item.get("image", None)
            or item.get("image_path", None)
            or item.get("img", None)
            or item.get("path", None)
            or item.get("image_file", None)
            or item.get("filename", None)
        )

        text = (
            item.get("text", "")
            or item.get("sentence", "")
            or item.get("caption", "")
            or item.get("content", "")
        )

        # ------------------------------------------------------------
        # Modality control
        # ------------------------------------------------------------
        if self.mode in ["image", "both"]:
            image = self._load_image(image_path)
        else:
            image = torch.zeros(3, 224, 224)

        if self.mode in ["text", "both"]:
            input_ids, attention_mask = self._encode_text(text)
        else:
            input_ids = torch.zeros(self.max_text_len, dtype=torch.long)
            attention_mask = torch.zeros(self.max_text_len, dtype=torch.long)

        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(label, dtype=torch.long),
        }


# ============================================================
# Backward-compatible aliases
# ============================================================

MVSADataset = MVSAStrongDataset
MVSAVoteDataset = MVSAStrongDataset
MVSA6ClassDataset = MVSAStrongDataset
VoteDataset = MVSAStrongDataset


# ============================================================
# Backward-compatible loader function
# ============================================================

def load_mvsa_datasets(
    train_json=None,
    val_json=None,
    test_json=None,
    tokenizer=None,
    mode="both",
    max_text_len=64,
    image_transform=image_transform,
    **kwargs,
):
    """
    Old compatibility function.

    Returns train/val/test datasets when json paths are given.

    Supports both:
        load_mvsa_datasets(train_json, val_json, test_json, tokenizer=...)
    and keyword-style calls.
    """

    # Also support alternative keyword names
    train_json = (
        train_json
        or kwargs.get("train_path", None)
        or kwargs.get("train_file", None)
    )

    val_json = (
        val_json
        or kwargs.get("val_path", None)
        or kwargs.get("valid_json", None)
        or kwargs.get("val_file", None)
    )

    test_json = (
        test_json
        or kwargs.get("test_path", None)
        or kwargs.get("test_file", None)
    )

    datasets = []

    for path in [train_json, val_json, test_json]:
        if path is None:
            datasets.append(None)
        else:
            datasets.append(
                MVSAStrongDataset(
                    json_path=path,
                    tokenizer=tokenizer,
                    mode=mode,
                    max_text_len=max_text_len,
                    image_transform=image_transform,
                )
            )

    return tuple(datasets)