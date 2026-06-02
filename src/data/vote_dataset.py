# ============================================================
# PyTorch Dataset for MVSA 6-Class Multimodal Sentiment Data
# ============================================================

import json

import torch
from torch.utils.data import Dataset
from PIL import Image


class MVSAVoteDataset(Dataset):
    """
    Dataset for MVSA 6-class multimodal sentiment classification.

    Each sample contains:
        image path
        text
        label
        label_name
    """

    def __init__(
        self,
        json_path,
        tokenizer=None,
        image_transform=None,
        max_length=128,
        return_metadata=False,
    ):
        """
        Args:
            json_path: path to mvsa_6class_train/val/test.json
            tokenizer: HuggingFace tokenizer, optional
            image_transform: torchvision image transform, optional
            max_length: max text length for tokenizer
            return_metadata: whether to return id, label_name, score
        """
        self.json_path = json_path
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_length = max_length
        self.return_metadata = return_metadata

        with open(json_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)

        if len(self.samples) == 0:
            raise ValueError(f"No samples found in {json_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        image_path = item["image"]
        text = item["text"]
        label = int(item["label"])

        # 读取图片：这里只在需要这个样本时才读取，不会一次性把全部图片加载进内存
        image = Image.open(image_path).convert("RGB")

        if self.image_transform is not None:
            image = self.image_transform(image)

        # 处理文本：如果传入 tokenizer，就转成 BERT/Transformer 输入格式
        if self.tokenizer is not None:
            text_inputs = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            # 去掉 tokenizer 自动加上的 batch 维度
            text_inputs = {
                key: value.squeeze(0)
                for key, value in text_inputs.items()
            }
        else:
            # 如果暂时不传 tokenizer，就直接返回原始文本
            text_inputs = text

        output = {
            "image": image,
            "text": text_inputs,
            "label": torch.tensor(label, dtype=torch.long),
        }

        if self.return_metadata:
            output["id"] = item.get("id", "")
            output["label_name"] = item.get("label_name", "")
            output["score"] = item.get("score", None)
            output["num_votes"] = item.get("num_votes", None)

        return output


def load_mvsa_datasets(
    train_json="data/processed/mvsa_6class_train.json",
    val_json="data/processed/mvsa_6class_val.json",
    test_json="data/processed/mvsa_6class_test.json",
    tokenizer=None,
    image_transform=None,
    max_length=128,
):
    """
    Load train, val, and test datasets.
    """

    train_dataset = MVSAVoteDataset(
        json_path=train_json,
        tokenizer=tokenizer,
        image_transform=image_transform,
        max_length=max_length,
    )

    val_dataset = MVSAVoteDataset(
        json_path=val_json,
        tokenizer=tokenizer,
        image_transform=image_transform,
        max_length=max_length,
    )

    test_dataset = MVSAVoteDataset(
        json_path=test_json,
        tokenizer=tokenizer,
        image_transform=image_transform,
        max_length=max_length,
    )

    return train_dataset, val_dataset, test_dataset