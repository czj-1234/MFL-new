# ============================================================
# Generic Multimodal Dataset
# Compatible with MVSA / Hateful Memes
#
# Supports:
#   1. json_path input
#   2. Python list input
#   3. image / text / both modality modes
#   4. CLIP image pixel_values format
#   5. RoBERTa / BERT tokenizer output
#   6. old imports: MVSAVoteDataset, load_mvsa_datasets
#
# Label logic:
#   - If item["label"] exists, use item["label"] directly.
#   - If item["label"] does not exist:
#       mode="text"  -> use item["text_label"]
#       mode="image" -> use item["image_label"]
#       mode="both"  -> use item["text_label"] by default
#
# Hateful Memes processed format:
#   label       = 0/1
#   text_label  = label
#   image_label = label
#
# MVSA processed format:
#   text_label and image_label may be different.
# ============================================================

import json
from pathlib import Path

from PIL import Image

import torch
from torch.utils.data import Dataset

try:
    from transformers import CLIPImageProcessor
except Exception:
    CLIPImageProcessor = None


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
    Generic multimodal dataset.

    Supports:
        data=list
        json_path=str
        first positional argument as either list or json_path

    Output keys:
        pixel_values
        image
        input_ids
        attention_mask
        label

    Label logic:
        1. If label_source="label", use item["label"].
        2. If label_source="text", use item["text_label"].
        3. If label_source="image", use item["image_label"].
        4. If label_source="auto":
            - if item["label"] exists, use it directly
            - else mode="text"  -> item["text_label"]
            - else mode="image" -> item["image_label"]
            - else mode="both"  -> item["text_label"] by default
    """

    def __init__(
        self,
        data=None,
        tokenizer=None,
        mode="both",
        max_text_len=64,
        json_path=None,
        image_model_name="openai/clip-vit-base-patch32",
        image_processor=None,
        transform=None,
        image_transform=None,
        max_len=None,
        max_length=None,
        label_source="auto",
        **kwargs,
    ):
        super().__init__()

        if isinstance(data, (str, Path)):
            json_path = data
            data = None

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

        if label_source not in ["auto", "label", "text", "image"]:
            raise ValueError(
                f"Unknown label_source: {label_source}. "
                "Expected auto, label, text, or image."
            )

        self.label_source = label_source

        if max_len is not None:
            max_text_len = max_len
        if max_length is not None:
            max_text_len = max_length

        self.max_text_len = int(max_text_len)

        if image_processor is not None:
            self.image_processor = image_processor
        else:
            if CLIPImageProcessor is None:
                raise ImportError(
                    "CLIPImageProcessor is not available. "
                    "Please install transformers: pip install transformers"
                )

            self.image_processor = CLIPImageProcessor.from_pretrained(
                image_model_name
            )

    def __len__(self):
        return len(self.data)

    def _resolve_image_path(self, image_path):
        if image_path is None:
            return None

        image_path = str(image_path).strip()
        image_path = image_path.replace("\\", "/")

        if image_path == "":
            return None

        p = Path(image_path)

        if p.exists():
            return p

        p2 = Path.cwd() / image_path

        if p2.exists():
            return p2

        candidates = [
            Path.cwd() / "data" / image_path,
            Path.cwd() / "data" / "raw" / image_path,
            Path.cwd() / "data" / "processed" / image_path,
            Path.cwd() / "data" / "images" / image_path,
            Path.cwd() / "data" / "MVSA" / image_path,
            Path.cwd() / "data" / "hateful_memes" / image_path,
            Path.cwd() / "data" / "Hateful Meme" / "hateful_memes" / image_path,
        ]

        for c in candidates:
            if c.exists():
                return c

        return p

    def _load_image(self, image_path):
        """
        Load image and return CLIP pixel_values tensor [3, 224, 224].
        If loading fails, return zero tensor.
        """
        try:
            resolved_path = self._resolve_image_path(image_path)

            if resolved_path is None:
                return torch.zeros(3, 224, 224)

            image = Image.open(resolved_path).convert("RGB")

            pixel_values = self.image_processor(
                images=image,
                return_tensors="pt",
            )["pixel_values"].squeeze(0)

            return pixel_values

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

    def _get_label(self, item):
        """
        Get label for the current sample.
        """

        if self.label_source == "label":
            if "label" not in item:
                raise KeyError(
                    "label_source='label' but item does not contain key 'label'."
                )
            return int(item["label"])

        if self.label_source == "text":
            if "text_label" not in item:
                raise KeyError(
                    "label_source='text' but item does not contain key 'text_label'."
                )
            return int(item["text_label"])

        if self.label_source == "image":
            if "image_label" not in item:
                raise KeyError(
                    "label_source='image' but item does not contain key 'image_label'."
                )
            return int(item["image_label"])

        # auto mode
        if "label" in item:
            return int(item["label"])

        if self.mode == "text":
            if "text_label" not in item:
                raise KeyError(
                    "mode='text' but item does not contain key 'text_label' "
                    "and no fixed key 'label' exists."
                )
            return int(item["text_label"])

        if self.mode == "image":
            if "image_label" not in item:
                raise KeyError(
                    "mode='image' but item does not contain key 'image_label' "
                    "and no fixed key 'label' exists."
                )
            return int(item["image_label"])

        if "text_label" in item:
            return int(item["text_label"])

        if "image_label" in item:
            return int(item["image_label"])

        raise KeyError(
            "Cannot find a valid label. Expected one of: "
            "'label', 'text_label', or 'image_label'."
        )

    def __getitem__(self, idx):
        item = self.data[idx]

        label = self._get_label(item)

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

        if self.mode in ["image", "both"]:
            pixel_values = self._load_image(image_path)
        else:
            pixel_values = torch.zeros(3, 224, 224)

        if self.mode in ["text", "both"]:
            input_ids, attention_mask = self._encode_text(text)
        else:
            input_ids = torch.zeros(self.max_text_len, dtype=torch.long)
            attention_mask = torch.zeros(self.max_text_len, dtype=torch.long)

        return {
            "pixel_values": pixel_values,

            # Keep old key for compatibility.
            "image": pixel_values,

            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(label, dtype=torch.long),
        }


# ============================================================
# Backward-compatible aliases
# ============================================================

MVSADataset = MVSAStrongDataset
MVSAVoteDataset = MVSAStrongDataset
MVSA4ClassDataset = MVSAStrongDataset
MVSA3ClassDataset = MVSAStrongDataset
MVSA6ClassDataset = MVSAStrongDataset
VoteDataset = MVSAStrongDataset

# Hateful Memes aliases
HatefulMemesDataset = MVSAStrongDataset
HatefulDataset = MVSAStrongDataset


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
    image_model_name="openai/clip-vit-base-patch32",
    image_processor=None,
    label_source="auto",
    **kwargs,
):
    """
    Old compatibility function.

    Returns train/val/test datasets when json paths are given.

    For modality-exclusive setting:
        text client:
            mode="text", label_source="text"
        image client:
            mode="image", label_source="image"

    If the input data has already been converted into single-modality samples
    with a fixed key "label", then label_source="auto" is enough.
    """

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
                    image_model_name=image_model_name,
                    image_processor=image_processor,
                    label_source=label_source,
                )
            )

    return tuple(datasets)


# Optional alias with a more general name
def load_multimodal_datasets(*args, **kwargs):
    return load_mvsa_datasets(*args, **kwargs)