# ============================================================
# Federated Learning Pipeline - Score Regression Variant
# Supports:
#   1. fixed-size client sampling
#   2. full-data client partitioning
#   3. image_only / text_only / modality_exclusive / full_multimodal
#   4. utility metrics: acc, macro-F1, precision, recall, balanced acc
#   5. structure / attack metrics
#   6. RoBERTa + CLIP-ViT-B/32 model input
# ============================================================

import os
import copy
import json
import random
import inspect
import tempfile
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import AutoTokenizer

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
)

import src.data.vote_dataset as vote_dataset

from src.metrics import compute_structure_metrics
from src.utils import load_json


# ============================================================
# Dataset Compatibility
# ============================================================

_DATASET_CANDIDATES = [
    "MVSAStrongDataset",
    "MVSADataset",
    "MVSAVoteDataset",
    "MVSA6ClassDataset",
    "VoteDataset",
]

MVSAStrongDataset = None

for _name in _DATASET_CANDIDATES:
    if hasattr(vote_dataset, _name):
        MVSAStrongDataset = getattr(vote_dataset, _name)
        break

if MVSAStrongDataset is None:
    dataset_like_names = [
        name for name in dir(vote_dataset)
        if "Dataset" in name
    ]

    if len(dataset_like_names) == 1:
        MVSAStrongDataset = getattr(vote_dataset, dataset_like_names[0])
    else:
        raise ImportError(
            "Cannot find a dataset class in src.data.vote_dataset. "
            f"Available Dataset-like names: {dataset_like_names}"
        )


image_transform = getattr(vote_dataset, "image_transform", None)


def build_mvsa_dataset(
    samples,
    tokenizer,
    mode,
    max_text_len,
    cache_dir=None,
    image_model_name="openai/clip-vit-base-patch32",
):
    """
    Build MVSA dataset while being compatible with different constructor styles.

    Supports both styles:
    1) Dataset(data=<list>, tokenizer=...) or Dataset(samples=<list>, tokenizer=...)
    2) Dataset(json_path=<path>, tokenizer=...) or Dataset(<json_path>, tokenizer=...)

    For CLIP-ViT image encoder, pass image_model_name to the Dataset when supported.
    """
    init_signature = inspect.signature(MVSAStrongDataset.__init__)
    params = init_signature.parameters
    valid_args = set(params.keys())
    param_names = [name for name in params.keys() if name != "self"]
    first_param = param_names[0] if len(param_names) > 0 else None

    kwargs = {}
    positional_arg = None

    # ------------------------------------------------------------
    # Optional arguments, only if supported by the Dataset class
    # ------------------------------------------------------------
    if "tokenizer" in valid_args:
        kwargs["tokenizer"] = tokenizer

    if "mode" in valid_args:
        kwargs["mode"] = mode

    if "max_text_len" in valid_args:
        kwargs["max_text_len"] = max_text_len

    if "max_len" in valid_args:
        kwargs["max_len"] = max_text_len

    if "max_length" in valid_args:
        kwargs["max_length"] = max_text_len

    if "image_model_name" in valid_args:
        kwargs["image_model_name"] = image_model_name

    # Keep old compatibility. New CLIP dataset may ignore these.
    if "image_transform" in valid_args:
        kwargs["image_transform"] = image_transform

    if "transform" in valid_args:
        kwargs["transform"] = image_transform

    # ------------------------------------------------------------
    # Data input compatibility
    # ------------------------------------------------------------
    if "data" in valid_args:
        kwargs["data"] = samples

    elif "samples" in valid_args:
        kwargs["samples"] = samples

    else:
        # Many older Dataset classes expect json_path as the first argument
        # and immediately call open(json_path). If samples is a list, create
        # a temporary JSON file and pass its path instead.
        path_like_names = {
            "json_path",
            "data_path",
            "file_path",
            "path",
            "json_file",
            "annotation_file",
            "ann_file",
        }

        expects_path = (
            first_param in path_like_names
            or any(name in valid_args for name in path_like_names)
        )

        if expects_path:
            if isinstance(samples, (str, bytes, os.PathLike)):
                json_path = samples
            else:
                if cache_dir is None:
                    cache_dir = tempfile.gettempdir()
                os.makedirs(cache_dir, exist_ok=True)

                tmp = tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".json",
                    prefix="mvsa_subset_",
                    dir=cache_dir,
                    delete=False,
                    encoding="utf-8",
                )
                with tmp:
                    json.dump(samples, tmp, ensure_ascii=False)
                json_path = tmp.name

            if "json_path" in valid_args:
                kwargs["json_path"] = json_path
            elif "data_path" in valid_args:
                kwargs["data_path"] = json_path
            elif "file_path" in valid_args:
                kwargs["file_path"] = json_path
            elif "path" in valid_args:
                kwargs["path"] = json_path
            elif "json_file" in valid_args:
                kwargs["json_file"] = json_path
            elif "annotation_file" in valid_args:
                kwargs["annotation_file"] = json_path
            elif "ann_file" in valid_args:
                kwargs["ann_file"] = json_path
            else:
                positional_arg = json_path

        else:
            # Last-resort fallback for Dataset(list, ...).
            positional_arg = samples

    # ------------------------------------------------------------
    # Constructor call
    # ------------------------------------------------------------
    if positional_arg is not None:
        return MVSAStrongDataset(positional_arg, **kwargs)

    return MVSAStrongDataset(**kwargs)


# ============================================================
# Label Mapping
# ============================================================

id_to_label_name = {
    0: "strong_negative",
    1: "weak_negative",
    2: "neutral_mixed",
    3: "weak_positive",
    4: "medium_positive",
    5: "strong_positive",
}


# ============================================================
# Regression Label Mapping
# ============================================================

label_id_to_score = {
    0: -2.5,  # strong_negative
    1: -1.5,  # weak_negative
    2:  0.0,  # neutral_mixed
    3:  1.0,  # weak_positive
    4:  1.8,  # medium_positive
    5:  2.5,  # strong_positive
}


def labels_to_scores(labels):
    """
    Convert class labels [0, 5] to continuous sentiment scores.
    """
    score_values = torch.tensor(
        [
            label_id_to_score[0],
            label_id_to_score[1],
            label_id_to_score[2],
            label_id_to_score[3],
            label_id_to_score[4],
            label_id_to_score[5],
        ],
        dtype=torch.float32,
        device=labels.device,
    )

    return score_values[labels.long()]


def scores_to_labels(scores):
    """
    Convert predicted sentiment scores back to 6-class labels.

    Thresholds are midpoints between neighboring sentiment scores:
        -2.5, -1.5, 0.0, 1.0, 1.8, 2.5
    """

    preds = torch.zeros_like(scores, dtype=torch.long)

    preds[(scores >= -2.0) & (scores < -0.75)] = 1
    preds[(scores >= -0.75) & (scores < 0.5)] = 2
    preds[(scores >= 0.5) & (scores < 1.4)] = 3
    preds[(scores >= 1.4) & (scores < 2.15)] = 4
    preds[scores >= 2.15] = 5

    return preds


# ============================================================
# Model Builder
# ============================================================

def build_model(args):
    """
    Build model from src.model.

    The project model.py already defines build_model(args), so we should use it
    directly. Do NOT call StrongMultimodalNet(args), because its first argument
    is text_model_name, not the whole args object.
    """
    import src.model as model_module

    if hasattr(model_module, "build_model"):
        return model_module.build_model(args)

    candidate_names = [
        "StrongMultimodalNet",
        "MVSAStrongModel",
        "StrongMultimodalModel",
        "MVSAFusionModel",
        "MultimodalFusionModel",
        "MultimodalSentimentModel",
        "MVSAClassifier",
        "MultimodalClassifier",
    ]

    model_cls = None

    for name in candidate_names:
        if hasattr(model_module, name):
            model_cls = getattr(model_module, name)
            break

    if model_cls is None:
        available = [
            name for name in dir(model_module)
            if not name.startswith("_")
        ]
        raise ImportError(
            "Could not find a supported model class in src/model.py. "
            f"Tried: {candidate_names}. "
            f"Available names: {available}"
        )

    image_model_name = getattr(
        args,
        "image_model_name",
        "openai/clip-vit-base-patch32",
    )

    return model_cls(
        text_model_name=args.text_model_name,
        image_model_name=image_model_name,
        num_classes=args.num_classes,
        image_hidden_dim=args.image_hidden_dim,
        text_hidden_dim=args.text_hidden_dim,
        projector_hidden_dim=args.projector_hidden_dim,
        dropout=args.dropout,
        freeze_image_backbone=args.freeze_image_backbone,
        freeze_text_backbone=args.freeze_text_backbone,
        pretrained_image=getattr(args, "pretrained_image", True),
    )


# ============================================================
# Client Modality / Label Assignment
# ============================================================

def get_client_modality(client_id, setting_name):
    """
    Return the modality available to each client.

    full_multimodal:
        all clients have both image and text

    image_only:
        all clients are image-only

    text_only:
        all clients are text-only

    modality_exclusive:
        even clients are image-only
        odd clients are text-only

    Important:
        The label space is always the same global 6-class task.
        Labels are not split by modality.
    """
    if setting_name == "full_multimodal":
        return "both"

    if setting_name == "image_only":
        return "image"

    if setting_name == "text_only":
        return "text"

    if setting_name == "modality_exclusive":
        return "image" if client_id % 2 == 0 else "text"

    raise ValueError(f"Unknown setting_name: {setting_name}")


def get_client_dominant_label(client_id, num_classes=6):
    """
    Assign one reference label to each client.

    In associated settings, this is the dominant label.
    In iid settings, this is only an assigned reference label.
    """
    return client_id % num_classes


# ============================================================
# Client Partitioning: Full-data Mode
# ============================================================

def build_full_client_partitions(
    train_data,
    setting_name,
    association,
    num_clients=6,
    num_classes=6,
    seed=42,
):
    """
    Build client partitions using the full training dataset.

    This mode uses every training sample exactly once.

    association:
        iid:
            Randomly split all training samples into clients.

        0.3 / 0.7 / 1.0:
            For each label, samples are assigned to the corresponding
            target client with probability equal to association.
            Remaining samples are distributed to other clients.

    Important:
        All settings still share the same global 6-class classification task.
        The label space is not split by modality.
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    labels = list(range(num_classes))

    client_data = {
        client_id: []
        for client_id in range(num_clients)
    }

    # ------------------------------------------------------------
    # Full-data IID split
    # ------------------------------------------------------------
    if association == "iid":
        all_samples = list(train_data)
        rng.shuffle(all_samples)

        splits = np.array_split(all_samples, num_clients)

        for client_id, split_samples in enumerate(splits):
            split_samples = list(split_samples)
            client_data[client_id] = split_samples

            modality = get_client_modality(client_id, setting_name)
            assigned_label = get_client_dominant_label(client_id, num_classes)

            print(
                f"Client {client_id} | "
                f"setting={setting_name} | "
                f"modality={modality} | "
                f"assigned_label={assigned_label} "
                f"({id_to_label_name.get(assigned_label, assigned_label)}) | "
                f"num_samples={len(split_samples)} | "
                f"label_dist={Counter([x['label_name'] for x in split_samples])}"
            )

        print("\nFull-data IID partition summary:")
        print("Total train samples:", len(train_data))
        print("Total assigned samples:", sum(len(v) for v in client_data.values()))

        return client_data

    # ------------------------------------------------------------
    # Full-data associated split
    # ------------------------------------------------------------
    association_strength = float(association)

    label_buckets = defaultdict(list)

    for item in train_data:
        label = int(item["label"])
        label_buckets[label].append(item)

    for label in labels:
        samples = label_buckets[label]
        rng.shuffle(samples)

        target_client = label % num_clients

        if association_strength >= 1.0:
            probs = np.zeros(num_clients, dtype=np.float64)
            probs[target_client] = 1.0
        else:
            probs = np.ones(num_clients, dtype=np.float64)
            probs[:] = (1.0 - association_strength) / (num_clients - 1)
            probs[target_client] = association_strength

        probs = probs / probs.sum()

        assigned_clients = np_rng.choice(
            np.arange(num_clients),
            size=len(samples),
            replace=True,
            p=probs,
        )

        for sample, client_id in zip(samples, assigned_clients):
            client_data[int(client_id)].append(sample)

    for client_id in range(num_clients):
        rng.shuffle(client_data[client_id])

        modality = get_client_modality(client_id, setting_name)
        dominant_label = get_client_dominant_label(client_id, num_classes)

        print(
            f"Client {client_id} | "
            f"setting={setting_name} | "
            f"modality={modality} | "
            f"dominant_label={dominant_label} "
            f"({id_to_label_name.get(dominant_label, dominant_label)}) | "
            f"num_samples={len(client_data[client_id])} | "
            f"label_dist={Counter([x['label_name'] for x in client_data[client_id]])}"
        )

    print("\nFull-data associated partition summary:")
    print("Total train samples:", len(train_data))
    print("Total assigned samples:", sum(len(v) for v in client_data.values()))
    print("Association:", association)

    return client_data


# ============================================================
# Client Partitioning: Fixed-size Mode
# ============================================================

def build_client_partitions(
    train_data,
    setting_name,
    association,
    num_clients=6,
    num_classes=6,
    samples_per_client=None,
    allow_overlap=False,
    seed=42,
    partition_mode="fixed",
):
    """
    Build client local datasets.

    partition_mode:
        fixed:
            Use fixed samples_per_client for each client.
            This is the old controlled experimental setting.

        full:
            Use the full training dataset exactly once.
            This is the full-data experimental setting.

    association:
        iid:
            balanced/fair split depending on partition mode

        0.3 / 0.7 / 1.0:
            dominant-label association strength
    """

    if partition_mode == "full":
        return build_full_client_partitions(
            train_data=train_data,
            setting_name=setting_name,
            association=association,
            num_clients=num_clients,
            num_classes=num_classes,
            seed=seed,
        )

    rng = random.Random(seed)

    if samples_per_client is None:
        samples_per_client = len(train_data) // num_clients

    label_buckets = defaultdict(list)

    for item in train_data:
        label = int(item["label"])
        label_buckets[label].append(item)

    for label in range(num_classes):
        rng.shuffle(label_buckets[label])

    available_buckets = {
        label: list(items)
        for label, items in label_buckets.items()
    }

    def sample_from_label(label, n):
        bucket = available_buckets[label]

        if len(bucket) == 0:
            raise ValueError(f"No samples available for label {label}.")

        if allow_overlap:
            if len(bucket) >= n:
                return rng.sample(bucket, n)

            return [rng.choice(bucket) for _ in range(n)]

        if len(bucket) < n:
            raise ValueError(
                f"Not enough samples for label {label}. "
                f"Need {n}, only {len(bucket)} available. "
                f"Set allow_overlap=True or reduce samples_per_client."
            )

        selected = bucket[:n]
        del bucket[:n]

        return selected

    client_data = {}

    for client_id in range(num_clients):
        dominant_label = get_client_dominant_label(client_id, num_classes)

        if association == "iid":
            base = samples_per_client // num_classes
            remainder = samples_per_client % num_classes

            label_counts = {label: base for label in range(num_classes)}

            extra_labels = list(range(num_classes))
            rng.shuffle(extra_labels)

            for label in extra_labels[:remainder]:
                label_counts[label] += 1

        else:
            assoc = float(association)

            dominant_count = int(round(samples_per_client * assoc))
            remaining = samples_per_client - dominant_count

            other_labels = [
                label for label in range(num_classes)
                if label != dominant_label
            ]

            base = remaining // len(other_labels)
            remainder = remaining % len(other_labels)

            label_counts = {label: 0 for label in range(num_classes)}
            label_counts[dominant_label] = dominant_count

            rng.shuffle(other_labels)

            for label in other_labels:
                label_counts[label] = base

            for label in other_labels[:remainder]:
                label_counts[label] += 1

        selected_samples = []

        for label, count in label_counts.items():
            if count > 0:
                selected_samples.extend(sample_from_label(label, count))

        rng.shuffle(selected_samples)
        client_data[client_id] = selected_samples

        print(
            f"Client {client_id} | "
            f"setting={setting_name} | "
            f"modality={get_client_modality(client_id, setting_name)} | "
            f"dominant_label={dominant_label} "
            f"({id_to_label_name.get(dominant_label, dominant_label)}) | "
            f"num_samples={len(selected_samples)} | "
            f"label_dist={Counter([x['label_name'] for x in selected_samples])}"
        )

    return client_data


# ============================================================
# DataLoader Helpers
# ============================================================

def build_dataloader(samples, tokenizer, args, mode, shuffle=True):
    dataset = build_mvsa_dataset(
        samples=samples,
        tokenizer=tokenizer,
        mode=mode,
        max_text_len=args.max_text_len,
        cache_dir=os.path.join(args.out_dir, "_dataset_cache"),
        image_model_name=getattr(
            args,
            "image_model_name",
            "openai/clip-vit-base-patch32",
        ),
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
    )

    return loader


def apply_modality_mask(batch, mode, device):
    """
    Apply modality masking at batch level.

    image:
        keep image, mask text into a constant empty-text input
    text:
        keep text, mask image into a zero CLIP pixel_values tensor
    both:
        keep both modalities

    Compatible with both old Dataset output:
        batch["image"]

    and new CLIP Dataset output:
        batch["pixel_values"]
    """

    if "pixel_values" in batch:
        image = batch["pixel_values"].to(device)
    else:
        image = batch["image"].to(device)

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["label"].to(device)

    if mode == "image":
        # Keep image branch. Mask text branch.
        input_ids = torch.zeros_like(input_ids)
        attention_mask = torch.zeros_like(attention_mask)

    elif mode == "text":
        # Keep text branch. Mask image branch.
        image = torch.zeros_like(image)

    elif mode == "both":
        pass

    else:
        raise ValueError(f"Unknown modality mode: {mode}")

    return image, input_ids, attention_mask, labels


# ============================================================
# Local Training
# ============================================================

def local_train(
    global_model,
    client_samples,
    tokenizer,
    args,
    mode,
):
    """
    Train one local client and return:
        local_state_dict
        update_vector
        local_loss
        local_acc
    """
    device = args.device

    local_model = copy.deepcopy(global_model)
    local_model.to(device)
    local_model.train()

    before_state = {
        k: v.detach().cpu().clone()
        for k, v in local_model.state_dict().items()
    }

    loader = build_dataloader(
        samples=client_samples,
        tokenizer=tokenizer,
        args=args,
        mode=mode,
        shuffle=True,
    )

    optimizer = torch.optim.AdamW(
        [p for p in local_model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=getattr(args, "weight_decay", 0.01),
    )

    total_loss = 0.0
    total = 0
    correct = 0
    step_count = 0

    for _ in range(args.local_epochs):
        for batch in loader:
            image, input_ids, attention_mask, labels = apply_modality_mask(
                batch=batch,
                mode=mode,
                device=device,
            )

            logits = local_model(
                image=image,
                input_ids=input_ids,
                attention_mask=attention_mask,
                setting=mode,
            )

            scores = logits.squeeze(-1)
            target_scores = labels_to_scores(labels)

            loss = F.smooth_l1_loss(
                scores,
                target_scores,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = scores_to_labels(scores)

            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            step_count += 1

            if args.max_local_steps is not None:
                if step_count >= args.max_local_steps:
                    break

        if args.max_local_steps is not None:
            if step_count >= args.max_local_steps:
                break

    after_state = {
        k: v.detach().cpu().clone()
        for k, v in local_model.state_dict().items()
    }

    update_vector = extract_update_vector(
        before_state=before_state,
        after_state=after_state,
        target_patterns=args.target_patterns,
    )

    avg_loss = total_loss / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0

    return after_state, update_vector, avg_loss, acc


# ============================================================
# Update Extraction
# ============================================================

def match_target_parameter(name, target_patterns):
    for pattern in target_patterns:
        if pattern in name:
            return True

    return False


def extract_update_vector(before_state, after_state, target_patterns):
    """
    Extract flattened parameter update vector for selected parameters.
    """
    vectors = []

    for name in before_state.keys():
        if match_target_parameter(name, target_patterns):
            diff = after_state[name] - before_state[name]
            vectors.append(diff.reshape(-1))

    if len(vectors) == 0:
        raise ValueError(
            "No parameters matched target_patterns. "
            f"target_patterns={target_patterns}"
        )

    return torch.cat(vectors).numpy()


# ============================================================
# FedAvg
# ============================================================

def fedavg_state_dicts(state_dicts, weights):
    """
    Weighted FedAvg over local state dicts.
    """
    total_weight = float(sum(weights))

    if total_weight <= 0:
        raise ValueError("Total FedAvg weight must be positive.")

    avg_state = {}

    for key in state_dicts[0].keys():
        avg_tensor = None

        for state, weight in zip(state_dicts, weights):
            tensor = state[key].float() * (weight / total_weight)

            if avg_tensor is None:
                avg_tensor = tensor
            else:
                avg_tensor += tensor

        avg_state[key] = avg_tensor

    return avg_state


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(model, data, tokenizer, args, mode="both", max_samples=None):
    """
    Evaluate score regression and convert scores back to 6-class labels.

    Returns:
        {
            "loss": float,
            "acc": float,
            "macro_f1": float,
            "macro_precision": float,
            "macro_recall": float,
            "balanced_acc": float,
        }
    """
    model.eval()
    model.to(args.device)

    if max_samples is not None and len(data) > max_samples:
        data = random.sample(data, max_samples)

    dataset = build_mvsa_dataset(
        samples=data,
        tokenizer=tokenizer,
        mode=mode,
        max_text_len=args.max_text_len,
        cache_dir=os.path.join(args.out_dir, "_dataset_cache"),
        image_model_name=getattr(
            args,
            "image_model_name",
            "openai/clip-vit-base-patch32",
        ),
    )

    loader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    total = 0
    correct = 0
    total_loss = 0.0

    all_labels = []
    all_preds = []

    for batch in loader:
        image, input_ids, attention_mask, labels = apply_modality_mask(
            batch=batch,
            mode=mode,
            device=args.device,
        )

        logits = model(
            image=image,
            input_ids=input_ids,
            attention_mask=attention_mask,
            setting=mode,
        )

        scores = logits.squeeze(-1)
        target_scores = labels_to_scores(labels)

        loss = F.smooth_l1_loss(
            scores,
            target_scores,
        )

        preds = scores_to_labels(scores)

        total += labels.size(0)
        correct += (preds == labels).sum().item()
        total_loss += loss.item() * labels.size(0)

        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_preds.extend(preds.detach().cpu().numpy().tolist())

    avg_loss = total_loss / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0

    if total > 0:
        macro_f1 = f1_score(
            all_labels,
            all_preds,
            average="macro",
            zero_division=0,
        )

        macro_precision = precision_score(
            all_labels,
            all_preds,
            average="macro",
            zero_division=0,
        )

        macro_recall = recall_score(
            all_labels,
            all_preds,
            average="macro",
            zero_division=0,
        )

        balanced_acc = balanced_accuracy_score(
            all_labels,
            all_preds,
        )
    else:
        macro_f1 = 0.0
        macro_precision = 0.0
        macro_recall = 0.0
        balanced_acc = 0.0

    return {
        "loss": float(avg_loss),
        "acc": float(acc),
        "macro_f1": float(macro_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "balanced_acc": float(balanced_acc),
    }


# ============================================================
# JSON Helper
# ============================================================

def make_json_serializable(obj):
    """
    Convert numpy / torch objects to JSON-serializable objects.
    """
    if isinstance(obj, dict):
        return {
            k: make_json_serializable(v)
            for k, v in obj.items()
        }

    if isinstance(obj, list):
        return [
            make_json_serializable(v)
            for v in obj
        ]

    if isinstance(obj, tuple):
        return tuple(
            make_json_serializable(v)
            for v in obj
        )

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()

    return obj


def save_summary_json(summary, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            make_json_serializable(summary),
            f,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# Main Experiment
# ============================================================

def run_experiment(args):
    """
    Run one federated learning experiment.
    """
    os.makedirs(args.out_dir, exist_ok=True)

    # ------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------
    print("\nLoading data...")

    train_data = load_json(args.train_json)
    val_data = load_json(args.val_json)
    test_data = load_json(args.test_json)

    print("Train:", len(train_data), Counter([x["label_name"] for x in train_data]))
    print("Val:  ", len(val_data), Counter([x["label_name"] for x in val_data]))
    print("Test: ", len(test_data), Counter([x["label_name"] for x in test_data]))

    # ------------------------------------------------------------
    # 2. Load tokenizer and model
    # ------------------------------------------------------------
    print("\nLoading tokenizer and model...")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    global_model = build_model(args)
    global_model.to(args.device)

    # ------------------------------------------------------------
    # 3. Build client partitions
    # ------------------------------------------------------------
    client_data = build_client_partitions(
        train_data=train_data,
        setting_name=args.setting_name,
        association=args.association,
        num_clients=args.num_clients,
        num_classes=args.num_classes,
        samples_per_client=args.samples_per_client,
        allow_overlap=args.allow_overlap,
        seed=args.seed,
        partition_mode=getattr(args, "partition_mode", "fixed"),
    )

    # ------------------------------------------------------------
    # Evaluation modality
    # ------------------------------------------------------------
    if args.setting_name == "image_only":
        eval_mode = "image"
    elif args.setting_name == "text_only":
        eval_mode = "text"
    else:
        eval_mode = "both"

    print(f"Evaluation mode: {eval_mode}")

    # ------------------------------------------------------------
    # 4. Federated training
    # ------------------------------------------------------------
    round_logs = []
    update_records = []
    update_metadata = []

    round_pbar = tqdm(
        range(1, args.rounds + 1),
        total=args.rounds,
        desc=f"{args.setting_name}-{args.association}",
        unit="round",
        dynamic_ncols=True,
    )

    for round_id in round_pbar:
        local_states = []
        local_weights = []
        round_local_losses = []
        round_local_accs = []

        for client_id in range(args.num_clients):
            samples = client_data[client_id]
            modality = get_client_modality(client_id, args.setting_name)
            dominant_label = get_client_dominant_label(
                client_id,
                args.num_classes,
            )

            local_state, update_vector, local_loss, local_acc = local_train(
                global_model=global_model,
                client_samples=samples,
                tokenizer=tokenizer,
                args=args,
                mode=modality,
            )

            round_local_losses.append(local_loss)
            round_local_accs.append(local_acc)

            local_states.append(local_state)
            local_weights.append(len(samples))

            update_records.append({
                "update": update_vector,
                "dominant_label": dominant_label,
            })

            update_metadata.append({
                "round": round_id,
                "client_id": client_id,
                "setting_name": args.setting_name,
                "association": args.association,
                "partition_mode": getattr(args, "partition_mode", "fixed"),
                "modality": modality,
                "dominant_label": dominant_label,
                "dominant_label_name": id_to_label_name.get(
                    dominant_label,
                    str(dominant_label),
                ),
                "num_samples": len(samples),
                "local_loss": local_loss,
                "local_acc": local_acc,
                "update_norm": float(np.linalg.norm(update_vector)),
            })

        avg_state = fedavg_state_dicts(
            state_dicts=local_states,
            weights=local_weights,
        )

        global_model.load_state_dict(avg_state, strict=True)

        avg_local_loss = (
            float(np.mean(round_local_losses))
            if len(round_local_losses) > 0
            else 0.0
        )

        avg_local_acc = (
            float(np.mean(round_local_accs))
            if len(round_local_accs) > 0
            else 0.0
        )

        # This is local client training performance for the current round.
        round_pbar.set_postfix({
            "local_loss": f"{avg_local_loss:.4f}",
            "local_acc": f"{avg_local_acc:.4f}",
        })

        # ------------------------------------------------------------
        # Global evaluation every 10 rounds
        # ------------------------------------------------------------
        if round_id == 1 or round_id % 10 == 0 or round_id == args.rounds:
            train_metrics = evaluate(
                global_model,
                train_data,
                tokenizer,
                args,
                mode=eval_mode,
                max_samples=args.max_train_eval_samples,
            )

            val_metrics = evaluate(
                global_model,
                val_data,
                tokenizer,
                args,
                mode=eval_mode,
                max_samples=args.max_val_eval_samples,
            )

            test_metrics = evaluate(
                global_model,
                test_data,
                tokenizer,
                args,
                mode=eval_mode,
                max_samples=args.max_test_eval_samples,
            )

            tqdm.write(
                f"\nRound {round_id:03d} Global Evaluation\n"
                f"Train | "
                f"Loss: {train_metrics['loss']:.4f} | "
                f"Acc: {train_metrics['acc']:.4f} | "
                f"Macro-F1: {train_metrics['macro_f1']:.4f}\n"
                f"Val   | "
                f"Loss: {val_metrics['loss']:.4f} | "
                f"Acc: {val_metrics['acc']:.4f} | "
                f"Macro-F1: {val_metrics['macro_f1']:.4f}\n"
                f"Test  | "
                f"Loss: {test_metrics['loss']:.4f} | "
                f"Acc: {test_metrics['acc']:.4f} | "
                f"Macro-F1: {test_metrics['macro_f1']:.4f}\n"
            )

            round_logs.append({
                "round": round_id,

                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["acc"],
                "train_macro_f1": train_metrics["macro_f1"],
                "train_macro_precision": train_metrics["macro_precision"],
                "train_macro_recall": train_metrics["macro_recall"],
                "train_balanced_acc": train_metrics["balanced_acc"],

                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_macro_precision": val_metrics["macro_precision"],
                "val_macro_recall": val_metrics["macro_recall"],
                "val_balanced_acc": val_metrics["balanced_acc"],

                "test_loss": test_metrics["loss"],
                "test_acc": test_metrics["acc"],
                "test_macro_f1": test_metrics["macro_f1"],
                "test_macro_precision": test_metrics["macro_precision"],
                "test_macro_recall": test_metrics["macro_recall"],
                "test_balanced_acc": test_metrics["balanced_acc"],
            })

    # ------------------------------------------------------------
    # 5. Final evaluation
    # ------------------------------------------------------------
    final_train_metrics = evaluate(
        global_model,
        train_data,
        tokenizer,
        args,
        mode=eval_mode,
        max_samples=args.max_train_eval_samples,
    )

    final_val_metrics = evaluate(
        global_model,
        val_data,
        tokenizer,
        args,
        mode=eval_mode,
        max_samples=args.max_val_eval_samples,
    )

    final_test_metrics = evaluate(
        global_model,
        test_data,
        tokenizer,
        args,
        mode=eval_mode,
        max_samples=args.max_test_eval_samples,
    )

    # ------------------------------------------------------------
    # 6. Structure and attack metrics
    # ------------------------------------------------------------
    structure_metrics, all_mat = compute_structure_metrics(
        update_records,
        seed=args.seed,
    )

    # ------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------
    global_num_classes = args.num_classes
    random_chance_acc = 1.0 / global_num_classes

    summary = {
        "setting_name": args.setting_name,
        "association": args.association,
        "num_clients": args.num_clients,
        "samples_per_client": args.samples_per_client,
        "partition_mode": getattr(args, "partition_mode", "fixed"),
        "allow_overlap": args.allow_overlap,
        "rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "lr": args.lr,
        "seed": args.seed,
        "model": "CLIP-ViT-B32+RoBERTa-base-regression",
        "freeze_image_backbone": args.freeze_image_backbone,
        "freeze_text_backbone": args.freeze_text_backbone,

        # task definition
        "task_type": "score_regression_then_6class_classification",
        "global_num_classes": global_num_classes,
        "random_chance_acc": random_chance_acc,
        "label_space_split_by_modality": False,
        "regression_score_mapping": {
            "strong_negative": -2.5,
            "weak_negative": -1.5,
            "neutral_mixed": 0.0,
            "weak_positive": 1.0,
            "medium_positive": 1.8,
            "strong_positive": 2.5,
        },

        # train utility metrics
        "train_loss": final_train_metrics["loss"],
        "train_acc": final_train_metrics["acc"],
        "train_macro_f1": final_train_metrics["macro_f1"],
        "train_macro_precision": final_train_metrics["macro_precision"],
        "train_macro_recall": final_train_metrics["macro_recall"],
        "train_balanced_acc": final_train_metrics["balanced_acc"],

        # validation utility metrics
        "val_loss": final_val_metrics["loss"],
        "val_acc": final_val_metrics["acc"],
        "val_macro_f1": final_val_metrics["macro_f1"],
        "val_macro_precision": final_val_metrics["macro_precision"],
        "val_macro_recall": final_val_metrics["macro_recall"],
        "val_balanced_acc": final_val_metrics["balanced_acc"],

        # global means validation on the shared global 6-class task
        "global_acc": final_val_metrics["acc"],
        "global_macro_f1": final_val_metrics["macro_f1"],
        "global_macro_precision": final_val_metrics["macro_precision"],
        "global_macro_recall": final_val_metrics["macro_recall"],
        "global_balanced_acc": final_val_metrics["balanced_acc"],

        # test utility metrics
        "test_loss": final_test_metrics["loss"],
        "test_acc": final_test_metrics["acc"],
        "test_macro_f1": final_test_metrics["macro_f1"],
        "test_macro_precision": final_test_metrics["macro_precision"],
        "test_macro_recall": final_test_metrics["macro_recall"],
        "test_balanced_acc": final_test_metrics["balanced_acc"],

        # accuracy above random chance
        "train_acc_above_chance": final_train_metrics["acc"] - random_chance_acc,
        "val_acc_above_chance": final_val_metrics["acc"] - random_chance_acc,
        "global_acc_above_chance": final_val_metrics["acc"] - random_chance_acc,
        "test_acc_above_chance": final_test_metrics["acc"] - random_chance_acc,
    }

    summary.update(structure_metrics)

    # ------------------------------------------------------------
    # 8. Save results
    # ------------------------------------------------------------
    summary_path = os.path.join(args.out_dir, "summary.json")
    round_logs_path = os.path.join(args.out_dir, "round_logs.csv")
    update_metadata_path = os.path.join(args.out_dir, "update_metadata.csv")
    update_matrix_path = os.path.join(args.out_dir, "update_matrix.npy")

    save_summary_json(summary, summary_path)
    pd.DataFrame(round_logs).to_csv(round_logs_path, index=False)
    pd.DataFrame(update_metadata).to_csv(update_metadata_path, index=False)
    np.save(update_matrix_path, all_mat)

    print("\nSaved results:")
    print("summary:", summary_path)
    print("round_logs:", round_logs_path)
    print("update_metadata:", update_metadata_path)
    print("update_matrix:", update_matrix_path)

    return summary, round_logs, all_mat