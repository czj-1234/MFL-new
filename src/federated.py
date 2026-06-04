# ============================================================
# Federated Learning Pipeline
# Supports:
#   1. fixed-size client sampling
#   2. full-data client partitioning
#   3. image_only / text_only / modality_exclusive / full_multimodal
#   4. utility metrics: acc, macro-F1, precision, recall, balanced acc
#   5. structure / attack metrics
#   6. RoBERTa + CLIP-ViT-B/32 model input
#   7. MVSA original 3-class classification
#
# Important for your current setting:
#   text_only:
#       client 0/1/2 = text, label = text_label
#
#   image_only:
#       client 0/1/2 = image, label = image_label
#
#   modality_exclusive:
#       client 0 = text,  dominant_label = negative
#       client 1 = text,  dominant_label = neutral
#       client 2 = image, dominant_label = positive
#
#   association is constructed using the label of the client's modality:
#       text client  -> text_label
#       image client -> image_label
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
    "MVSA3ClassDataset",
    "MVSA4ClassDataset",
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
    label_source="auto",
):
    """
    Build MVSA dataset while being compatible with different constructor styles.

    Supports both:
    1) Dataset(data=<list>, tokenizer=...)
    2) Dataset(json_path=<path>, tokenizer=...)

    For the new 3-class data:
        label_source="text"  -> use text_label
        label_source="image" -> use image_label
        label_source="auto"  -> use item["label"] if already fixed
    """
    init_signature = inspect.signature(MVSAStrongDataset.__init__)
    params = init_signature.parameters
    valid_args = set(params.keys())
    param_names = [name for name in params.keys() if name != "self"]
    first_param = param_names[0] if len(param_names) > 0 else None

    kwargs = {}
    positional_arg = None

    # ------------------------------------------------------------
    # Optional arguments
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

    if "label_source" in valid_args:
        kwargs["label_source"] = label_source

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
            positional_arg = samples

    if positional_arg is not None:
        return MVSAStrongDataset(positional_arg, **kwargs)

    return MVSAStrongDataset(**kwargs)


# ============================================================
# Label Mapping
# ============================================================

id_to_label_name = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


# ============================================================
# Model Builder
# ============================================================

def build_model(args):
    """
    Build model from src.model.
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

    image_only:
        all clients are image-only

    text_only:
        all clients are text-only

    modality_exclusive:
        fixed as:
            client 0 -> text
            client 1 -> text
            client 2 -> image

    full_multimodal:
        all clients have both image and text
    """
    if setting_name == "full_multimodal":
        return "both"

    if setting_name == "image_only":
        return "image"

    if setting_name == "text_only":
        return "text"

    if setting_name == "modality_exclusive":
        if client_id in [0, 1]:
            return "text"
        if client_id == 2:
            return "image"

        # Fallback if you later use more than 3 clients.
        return "text" if client_id % 3 in [0, 1] else "image"

    raise ValueError(f"Unknown setting_name: {setting_name}")


def get_client_dominant_label(client_id, num_classes=3):
    """
    Assign one dominant label to each client.

    For 3 clients and 3 classes:
        client 0 -> 0 negative
        client 1 -> 1 neutral
        client 2 -> 2 positive
    """
    return client_id % num_classes


def get_sample_label_for_modality(item, modality):
    """
    Get the label used by a specific modality.

    text client:
        use text_label

    image client:
        use image_label

    both:
        if item has fixed label, use it.
        otherwise use text_label by default.
    """
    if modality == "text":
        if "text_label" in item:
            return int(item["text_label"])
        if "label" in item:
            return int(item["label"])
        raise KeyError("Text sample has neither text_label nor label.")

    if modality == "image":
        if "image_label" in item:
            return int(item["image_label"])
        if "label" in item:
            return int(item["label"])
        raise KeyError("Image sample has neither image_label nor label.")

    if modality == "both":
        if "label" in item:
            return int(item["label"])
        if "text_label" in item:
            return int(item["text_label"])
        if "image_label" in item:
            return int(item["image_label"])
        raise KeyError("Both-modality sample has no valid label.")

    raise ValueError(f"Unknown modality: {modality}")


def get_sample_label_name_for_modality(item, modality):
    label = get_sample_label_for_modality(item, modality)
    return id_to_label_name.get(label, str(label))


def convert_to_client_sample(item, modality):
    """
    Convert original MVSA sample into a single-modality client sample
    with a fixed key: item["label"].

    This avoids confusion:
        text-view sample  -> label = text_label
        image-view sample -> label = image_label
    """
    label = get_sample_label_for_modality(item, modality)
    label_name = id_to_label_name.get(label, str(label))

    new_item = dict(item)
    new_item["modality"] = modality
    new_item["label"] = int(label)
    new_item["label_name"] = label_name

    if modality == "text":
        # Text client should not use image.
        # Keep image path for compatibility, but Dataset mode="text"
        # will mask image anyway.
        new_item["client_label_source"] = "text_label"

    elif modality == "image":
        # Image client should not use text.
        # Keep text for compatibility, but Dataset mode="image"
        # will mask text anyway.
        new_item["client_label_source"] = "image_label"

    elif modality == "both":
        new_item["client_label_source"] = "label_or_text_label"

    else:
        raise ValueError(f"Unknown modality: {modality}")

    return new_item


def get_eval_mode_and_label_source(setting_name):
    """
    Evaluation mode for a setting.

    For text_only:
        evaluate text branch with text_label

    For image_only:
        evaluate image branch with image_label

    For full_multimodal:
        evaluate both branches. If no unified label exists, Dataset falls back
        to text_label.

    For modality_exclusive:
        there is no single natural unified label because text clients and image
        clients use different modality-specific labels. We handle it separately
        in evaluate_for_setting().
    """
    if setting_name == "text_only":
        return "text", "text"

    if setting_name == "image_only":
        return "image", "image"

    if setting_name == "full_multimodal":
        return "both", "auto"

    if setting_name == "modality_exclusive":
        return None, None

    raise ValueError(f"Unknown setting_name: {setting_name}")


# ============================================================
# Class Weight Helper
# ============================================================

def compute_class_weights_from_client_data(client_data, num_classes, device):
    """
    Compute class weights from the actually used client samples.

    Since client samples already have a fixed key "label", this works for:
        text_only
        image_only
        modality_exclusive
    """
    labels = []

    for samples in client_data.values():
        for item in samples:
            labels.append(int(item["label"]))

    counts = Counter(labels)
    total = sum(counts.values())

    weights = []

    for label in range(num_classes):
        count = counts.get(label, 0)

        if count <= 0:
            weights.append(0.0)
        else:
            weights.append(total / (num_classes * count))

    # Avoid zero class weight if a class is missing in a rare partition.
    non_zero = [w for w in weights if w > 0]
    fallback = float(np.mean(non_zero)) if len(non_zero) > 0 else 1.0
    weights = [w if w > 0 else fallback for w in weights]

    return torch.tensor(weights, dtype=torch.float32, device=device)


# ============================================================
# Client Partitioning: Full-data Mode
# ============================================================

def build_full_client_partitions(
    train_data,
    setting_name,
    association,
    num_clients=3,
    num_classes=3,
    seed=42,
):
    """
    Build client partitions using the full training dataset.

    This function returns client samples with a fixed "label" key.

    For association:
        iid:
            Randomly split all raw samples into clients, then convert each
            sample according to that client's modality.

        0.3 / 0.7 / 1.0:
            For each raw sample, calculate which clients are label-matched
            according to their own modality label.

            Example for modality_exclusive:
                client 0: text  + dominant negative
                client 1: text  + dominant neutral
                client 2: image + dominant positive

            A sample is more likely to be assigned to a client if:
                sample[text_label]  == client dominant label for text clients
                sample[image_label] == client dominant label for image clients
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    client_data = {
        client_id: []
        for client_id in range(num_clients)
    }

    client_modalities = {
        client_id: get_client_modality(client_id, setting_name)
        for client_id in range(num_clients)
    }

    client_dominant_labels = {
        client_id: get_client_dominant_label(client_id, num_classes)
        for client_id in range(num_clients)
    }

    # ------------------------------------------------------------
    # IID split
    # ------------------------------------------------------------
    if association == "iid":
        all_samples = list(train_data)
        rng.shuffle(all_samples)

        splits = np.array_split(all_samples, num_clients)

        for client_id, split_samples in enumerate(splits):
            modality = client_modalities[client_id]

            converted = [
                convert_to_client_sample(item, modality)
                for item in list(split_samples)
            ]

            client_data[client_id] = converted

            assigned_label = client_dominant_labels[client_id]

            print(
                f"Client {client_id} | "
                f"setting={setting_name} | "
                f"modality={modality} | "
                f"assigned_label={assigned_label} "
                f"({id_to_label_name.get(assigned_label, assigned_label)}) | "
                f"num_samples={len(converted)} | "
                f"label_dist={Counter([x['label_name'] for x in converted])}"
            )

        print("\nFull-data IID partition summary:")
        print("Total train samples:", len(train_data))
        print("Total assigned samples:", sum(len(v) for v in client_data.values()))

        return client_data

    # ------------------------------------------------------------
    # Associated split
    # ------------------------------------------------------------
    association_strength = float(association)

    for item in train_data:
        matched_clients = []

        for client_id in range(num_clients):
            modality = client_modalities[client_id]
            dominant_label = client_dominant_labels[client_id]
            sample_label = get_sample_label_for_modality(item, modality)

            if sample_label == dominant_label:
                matched_clients.append(client_id)

        if len(matched_clients) == 0:
            probs = np.ones(num_clients, dtype=np.float64) / num_clients

        else:
            probs = np.ones(num_clients, dtype=np.float64)

            non_matched = [
                cid for cid in range(num_clients)
                if cid not in matched_clients
            ]

            if association_strength >= 1.0:
                probs[:] = 0.0
                for cid in matched_clients:
                    probs[cid] = 1.0 / len(matched_clients)
            else:
                probs[:] = 0.0

                matched_mass = association_strength
                other_mass = 1.0 - association_strength

                for cid in matched_clients:
                    probs[cid] = matched_mass / len(matched_clients)

                if len(non_matched) > 0:
                    for cid in non_matched:
                        probs[cid] = other_mass / len(non_matched)
                else:
                    # If every client is matched, distribute uniformly.
                    probs[:] = 1.0 / num_clients

        probs = probs / probs.sum()

        selected_client = int(
            np_rng.choice(
                np.arange(num_clients),
                size=1,
                replace=True,
                p=probs,
            )[0]
        )

        modality = client_modalities[selected_client]
        client_sample = convert_to_client_sample(item, modality)
        client_data[selected_client].append(client_sample)

    for client_id in range(num_clients):
        rng.shuffle(client_data[client_id])

        modality = client_modalities[client_id]
        dominant_label = client_dominant_labels[client_id]

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
    num_clients=3,
    num_classes=3,
    samples_per_client=None,
    allow_overlap=False,
    seed=42,
    partition_mode="fixed",
):
    """
    Build client local datasets.

    partition_mode:
        full:
            Use the full training dataset exactly once.

        fixed:
            Use fixed samples_per_client for each client.

    Output:
        client_data[client_id] is a list of samples with fixed:
            item["label"]
            item["label_name"]
            item["modality"]
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

    client_data = {}

    # For fixed mode, sample independently per client according to
    # that client's modality-specific label buckets.
    for client_id in range(num_clients):
        modality = get_client_modality(client_id, setting_name)
        dominant_label = get_client_dominant_label(client_id, num_classes)

        label_buckets = defaultdict(list)

        for item in train_data:
            label = get_sample_label_for_modality(item, modality)
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
                raise ValueError(
                    f"No samples available for label {label} "
                    f"under modality={modality}."
                )

            if allow_overlap:
                if len(bucket) >= n:
                    return rng.sample(bucket, n)

                return [rng.choice(bucket) for _ in range(n)]

            if len(bucket) < n:
                raise ValueError(
                    f"Not enough samples for label {label} "
                    f"under modality={modality}. "
                    f"Need {n}, only {len(bucket)} available. "
                    f"Set allow_overlap=True or reduce samples_per_client."
                )

            selected = bucket[:n]
            del bucket[:n]
            return selected

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

        selected_raw_samples = []

        for label, count in label_counts.items():
            if count > 0:
                selected_raw_samples.extend(sample_from_label(label, count))

        rng.shuffle(selected_raw_samples)

        selected_samples = [
            convert_to_client_sample(item, modality)
            for item in selected_raw_samples
        ]

        client_data[client_id] = selected_samples

        print(
            f"Client {client_id} | "
            f"setting={setting_name} | "
            f"modality={modality} | "
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
        label_source="auto",
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
        keep image, mask text

    text:
        keep text, mask image

    both:
        keep both modalities
    """

    if "pixel_values" in batch:
        image = batch["pixel_values"].to(device)
    else:
        image = batch["image"].to(device)

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["label"].to(device)

    if mode == "image":
        input_ids = torch.zeros_like(input_ids)
        attention_mask = torch.zeros_like(attention_mask)

    elif mode == "text":
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
        weight_decay=getattr(args, "weight_decay", 0.0),
    )

    if hasattr(args, "class_weights_tensor"):
        class_weights_tensor = args.class_weights_tensor.to(device)
    elif hasattr(args, "class_weights"):
        class_weights_tensor = torch.tensor(
            args.class_weights,
            dtype=torch.float32,
            device=device,
        )
    else:
        class_weights_tensor = None

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

            loss = F.cross_entropy(
                logits,
                labels,
                weight=class_weights_tensor,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = torch.argmax(logits, dim=1)

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
def evaluate(
    model,
    data,
    tokenizer,
    args,
    mode="both",
    max_samples=None,
    label_source="auto",
):
    """
    Evaluate classification task.

    For 3-class MVSA:
        text evaluation:
            mode="text", label_source="text"

        image evaluation:
            mode="image", label_source="image"

        converted client samples:
            label_source="auto"
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
        label_source=label_source,
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

    if hasattr(args, "class_weights_tensor"):
        class_weights_tensor = args.class_weights_tensor.to(args.device)
    elif hasattr(args, "class_weights"):
        class_weights_tensor = torch.tensor(
            args.class_weights,
            dtype=torch.float32,
            device=args.device,
        )
    else:
        class_weights_tensor = None

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

        loss = F.cross_entropy(logits, labels, weight=class_weights_tensor)

        preds = torch.argmax(logits, dim=1)

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


def average_metric_dicts(metric_dicts):
    """
    Average a list of metric dictionaries.
    """
    if len(metric_dicts) == 0:
        return {
            "loss": 0.0,
            "acc": 0.0,
            "macro_f1": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "balanced_acc": 0.0,
        }

    keys = metric_dicts[0].keys()

    return {
        key: float(np.mean([m[key] for m in metric_dicts]))
        for key in keys
    }


def evaluate_for_setting(
    model,
    data,
    tokenizer,
    args,
    max_samples=None,
):
    """
    Evaluate according to the experiment setting.

    text_only:
        text input + text_label

    image_only:
        image input + image_label

    full_multimodal:
        both input. If no fixed label exists, Dataset falls back to text_label.

    modality_exclusive:
        evaluate both text-view and image-view, then average metrics.
        This avoids pretending that each raw MVSA pair has one fixed unified label.
    """
    setting_name = args.setting_name

    if setting_name == "text_only":
        return evaluate(
            model=model,
            data=data,
            tokenizer=tokenizer,
            args=args,
            mode="text",
            max_samples=max_samples,
            label_source="text",
        )

    if setting_name == "image_only":
        return evaluate(
            model=model,
            data=data,
            tokenizer=tokenizer,
            args=args,
            mode="image",
            max_samples=max_samples,
            label_source="image",
        )

    if setting_name == "full_multimodal":
        return evaluate(
            model=model,
            data=data,
            tokenizer=tokenizer,
            args=args,
            mode="both",
            max_samples=max_samples,
            label_source="auto",
        )

    if setting_name == "modality_exclusive":
        text_metrics = evaluate(
            model=model,
            data=data,
            tokenizer=tokenizer,
            args=args,
            mode="text",
            max_samples=max_samples,
            label_source="text",
        )

        image_metrics = evaluate(
            model=model,
            data=data,
            tokenizer=tokenizer,
            args=args,
            mode="image",
            max_samples=max_samples,
            label_source="image",
        )

        avg_metrics = average_metric_dicts([text_metrics, image_metrics])

        # Keep extra detail for logs if needed.
        avg_metrics["text_acc"] = text_metrics["acc"]
        avg_metrics["image_acc"] = image_metrics["acc"]
        avg_metrics["text_macro_f1"] = text_metrics["macro_f1"]
        avg_metrics["image_macro_f1"] = image_metrics["macro_f1"]

        return avg_metrics

    raise ValueError(f"Unknown setting_name: {setting_name}")


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

    print("Train:", len(train_data))
    print("  text :", Counter([x["text_label_name"] for x in train_data]))
    print("  image:", Counter([x["image_label_name"] for x in train_data]))

    print("Val:  ", len(val_data))
    print("  text :", Counter([x["text_label_name"] for x in val_data]))
    print("  image:", Counter([x["image_label_name"] for x in val_data]))

    print("Test: ", len(test_data))
    print("  text :", Counter([x["text_label_name"] for x in test_data]))
    print("  image:", Counter([x["image_label_name"] for x in test_data]))

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
    # Class imbalance handling
    # ------------------------------------------------------------
    # Compute class weights from the actually used client samples.
    # This is important because:
    #   text clients use text_label
    #   image clients use image_label
    args.class_weights_tensor = compute_class_weights_from_client_data(
        client_data=client_data,
        num_classes=args.num_classes,
        device=args.device,
    )

    args.class_weights = args.class_weights_tensor.detach().cpu().tolist()

    print("\nClass weights from actual client samples:")
    print(args.class_weights)

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
                "client_label_dist": dict(
                    Counter([x["label_name"] for x in samples])
                ),
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

        round_pbar.set_postfix({
            "local_loss": f"{avg_local_loss:.4f}",
            "local_acc": f"{avg_local_acc:.4f}",
        })

        # ------------------------------------------------------------
        # Global evaluation every 10 rounds
        # ------------------------------------------------------------
        if round_id == 1 or round_id % 10 == 0 or round_id == args.rounds:
            train_metrics = evaluate_for_setting(
                global_model,
                train_data,
                tokenizer,
                args,
                max_samples=args.max_train_eval_samples,
            )

            val_metrics = evaluate_for_setting(
                global_model,
                val_data,
                tokenizer,
                args,
                max_samples=args.max_val_eval_samples,
            )

            test_metrics = evaluate_for_setting(
                global_model,
                test_data,
                tokenizer,
                args,
                max_samples=args.max_test_eval_samples,
            )

            extra_line = ""

            if args.setting_name == "modality_exclusive":
                extra_line = (
                    f"Modality detail | "
                    f"Val text Acc: {val_metrics.get('text_acc', 0.0):.4f} | "
                    f"Val image Acc: {val_metrics.get('image_acc', 0.0):.4f} | "
                    f"Test text Acc: {test_metrics.get('text_acc', 0.0):.4f} | "
                    f"Test image Acc: {test_metrics.get('image_acc', 0.0):.4f}\n"
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
                f"{extra_line}"
            )

            round_log = {
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
            }

            # Save modality-specific details when available.
            for key in [
                "text_acc",
                "image_acc",
                "text_macro_f1",
                "image_macro_f1",
            ]:
                if key in train_metrics:
                    round_log[f"train_{key}"] = train_metrics[key]
                if key in val_metrics:
                    round_log[f"val_{key}"] = val_metrics[key]
                if key in test_metrics:
                    round_log[f"test_{key}"] = test_metrics[key]

            round_logs.append(round_log)

    # ------------------------------------------------------------
    # 5. Final evaluation
    # ------------------------------------------------------------
    final_train_metrics = evaluate_for_setting(
        global_model,
        train_data,
        tokenizer,
        args,
        max_samples=args.max_train_eval_samples,
    )

    final_val_metrics = evaluate_for_setting(
        global_model,
        val_data,
        tokenizer,
        args,
        max_samples=args.max_val_eval_samples,
    )

    final_test_metrics = evaluate_for_setting(
        global_model,
        test_data,
        tokenizer,
        args,
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
        "weight_decay": getattr(args, "weight_decay", 0.0),
        "seed": args.seed,
        "model": "CLIP-ViT-B32+RoBERTa-base",
        "freeze_image_backbone": args.freeze_image_backbone,
        "freeze_text_backbone": args.freeze_text_backbone,

        # task definition
        "task_type": "mvsa_original_3class_modality_specific_classification",
        "global_num_classes": global_num_classes,
        "random_chance_acc": random_chance_acc,
        "label_space_split_by_modality": False,
        "uses_modality_specific_labels": True,
        "text_client_label_source": "text_label",
        "image_client_label_source": "image_label",

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

        # global means validation metric
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

    # Add modality-specific summary metrics when available.
    for key in [
        "text_acc",
        "image_acc",
        "text_macro_f1",
        "image_macro_f1",
    ]:
        if key in final_train_metrics:
            summary[f"train_{key}"] = final_train_metrics[key]
        if key in final_val_metrics:
            summary[f"val_{key}"] = final_val_metrics[key]
        if key in final_test_metrics:
            summary[f"test_{key}"] = final_test_metrics[key]

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