# ============================================================
# Federated Learning Pipeline
# ============================================================

import os
import copy
import random
from collections import Counter, defaultdict

import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
)

import torchvision.transforms as T
from transformers import AutoTokenizer

from src.model import build_model
from src.metrics import compute_structure_metrics
from src.utils import (
    set_seed,
    load_json,
    save_json,
    label_distribution,
)


# ============================================================
# 1. Label mapping
# ============================================================

NUM_CLASSES = 6

id_to_label_name = {
    0: "strong_negative",
    1: "weak_negative",
    2: "neutral_mixed",
    3: "weak_positive",
    4: "medium_positive",
    5: "strong_positive",
}

label_name_to_id = {v: k for k, v in id_to_label_name.items()}


# ============================================================
# 2. Image transform
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
# 3. Dataset for FL training
# ============================================================

class MVSAStrongDataset(torch.utils.data.Dataset):
    """
    Dataset used in federated training.

    mode:
        image: image-only client
        text : text-only client
        both : both modalities for evaluation
    """

    def __init__(
        self,
        data,
        tokenizer,
        mode="both",
        max_text_len=64,
        image_transform=None,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_text_len = max_text_len
        self.image_transform = image_transform

    def __len__(self):
        return len(self.data)

    def _load_image(self, path):
        from PIL import Image

        try:
            image = Image.open(path).convert("RGB")

            if self.image_transform is not None:
                image = self.image_transform(image)

            return image

        except Exception:
            return torch.zeros(3, 224, 224)

    def __getitem__(self, idx):
        item = self.data[idx]

        label = int(item["label"])
        image_path = item.get("image", "")
        text = str(item.get("text", ""))

        if self.mode in ["image", "both"]:
            image = self._load_image(image_path)
        else:
            image = torch.zeros(3, 224, 224)

        if self.mode in ["text", "both"]:
            encoded = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_text_len,
                return_tensors="pt",
            )

            input_ids = encoded["input_ids"].squeeze(0)
            attention_mask = encoded["attention_mask"].squeeze(0)

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
# 4. Client partition
# ============================================================

def get_client_modality(client_id, setting_name):
    """
    setting_name:
        image_only: all clients are image-only
        text_only: all clients are text-only
        modality_exclusive: even clients image-only, odd clients text-only
    """

    if setting_name == "image_only":
        return "image"

    if setting_name == "text_only":
        return "text"

    if setting_name == "modality_exclusive":
        return "image" if client_id % 2 == 0 else "text"

    raise ValueError(f"Unknown setting_name: {setting_name}")


def get_client_dominant_label(client_id, num_classes=6):
    return client_id % num_classes


def build_client_partitions(
    train_data,
    setting_name,
    association,
    num_clients=6,
    num_classes=6,
    samples_per_client=None,
    allow_overlap=False,
    seed=42,
):
    """
    Build client local datasets.

    association:
        "iid": balanced label distribution for each client
        "0.3", "0.7", "1.0": dominant-label association strength

    samples_per_client:
        None means automatically use all train samples approximately evenly.
    """

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
            f"dominant_label={dominant_label} ({id_to_label_name[dominant_label]}) | "
            f"label_dist={Counter([x['label_name'] for x in selected_samples])}"
        )

    return client_data


# ============================================================
# 5. Training and evaluation
# ============================================================

def train_one_client(global_model, local_data, tokenizer, args, client_modality):
    local_model = copy.deepcopy(global_model)
    local_model.to(args.device)
    local_model.train()

    dataset = MVSAStrongDataset(
        local_data,
        tokenizer=tokenizer,
        mode=client_modality,
        max_text_len=args.max_text_len,
        image_transform=image_transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    optimizer = torch.optim.AdamW(
        [p for p in local_model.parameters() if p.requires_grad],
        lr=args.lr,
    )

    step_count = 0

    for _ in range(args.local_epochs):
        for batch in loader:
            image = batch["image"].to(args.device)
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            labels = batch["label"].to(args.device)

            optimizer.zero_grad()

            logits = local_model(image, input_ids, attention_mask)
            loss = F.cross_entropy(logits, labels)

            loss.backward()
            optimizer.step()

            step_count += 1

            if args.max_local_steps is not None and step_count >= args.max_local_steps:
                break

        if args.max_local_steps is not None and step_count >= args.max_local_steps:
            break

    return local_model


def fedavg(global_model, local_models, client_sizes):
    global_state = global_model.state_dict()
    total_size = sum(client_sizes)

    new_state = {}

    for key in global_state.keys():
        new_state[key] = sum(
            local_models[i].state_dict()[key].detach().cpu()
            * (client_sizes[i] / total_size)
            for i in range(len(local_models))
        )

    global_model.load_state_dict(new_state)

    return global_model


@torch.no_grad()
def evaluate(model, data, tokenizer, args, mode="both", max_samples=None):
    """
    Evaluate the global 6-class classification task.

    Returns:
        {
            "loss": float,
            "acc": float,
            "macro_f1": float,
            "macro_precision": float,
            "macro_recall": float,
            "balanced_acc": float,
        }

    Note:
        All settings share the same global 6-class label space.
        The label space is not split by modality.
    """
    model.eval()
    model.to(args.device)

    if max_samples is not None and len(data) > max_samples:
        data = random.sample(data, max_samples)

    dataset = MVSAStrongDataset(
        data,
        tokenizer=tokenizer,
        mode=mode,
        max_text_len=args.max_text_len,
        image_transform=image_transform,
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
        image = batch["image"].to(args.device)
        input_ids = batch["input_ids"].to(args.device)
        attention_mask = batch["attention_mask"].to(args.device)
        labels = batch["label"].to(args.device)

        logits = model(image, input_ids, attention_mask)
        loss = F.cross_entropy(logits, labels)

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


# ============================================================
# 6. Update extraction
# ============================================================

def extract_update_vector(global_before, local_after, target_patterns):
    before_state = global_before.state_dict()
    after_state = local_after.state_dict()

    vecs = []

    for name in before_state.keys():
        if any(pattern in name for pattern in target_patterns):
            delta = (
                after_state[name].detach().cpu().float()
                - before_state[name].detach().cpu().float()
            )
            vecs.append(delta.reshape(-1))

    if len(vecs) == 0:
        raise ValueError(f"No parameters matched target_patterns: {target_patterns}")

    return torch.cat(vecs).numpy()


# ============================================================
# 7. Main experiment function
# ============================================================

def run_experiment(args):
    os.makedirs(args.out_dir, exist_ok=True)

    set_seed(args.seed)

    print("\nLoading data...")
    train_data = load_json(args.train_json)
    val_data = load_json(args.val_json)
    test_data = load_json(args.test_json)

    print("Train:", len(train_data), label_distribution(train_data))
    print("Val:  ", len(val_data), label_distribution(val_data))
    print("Test: ", len(test_data), label_distribution(test_data))

    print("\nLoading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    client_data = build_client_partitions(
        train_data=train_data,
        setting_name=args.setting_name,
        association=args.association,
        num_clients=args.num_clients,
        num_classes=args.num_classes,
        samples_per_client=args.samples_per_client,
        allow_overlap=args.allow_overlap,
        seed=args.seed,
    )

    global_model = build_model(args).to(args.device)

    update_records = []
    round_logs = []

    for round_id in range(1, args.rounds + 1):
        global_before = copy.deepcopy(global_model).cpu()

        local_models = []
        client_sizes = []

        for client_id in range(args.num_clients):
            modality = get_client_modality(client_id, args.setting_name)
            dominant_label = get_client_dominant_label(client_id, args.num_classes)

            local_model = train_one_client(
                global_model=global_model,
                local_data=client_data[client_id],
                tokenizer=tokenizer,
                args=args,
                client_modality=modality,
            )

            update_vec = extract_update_vector(
                global_before=global_before,
                local_after=local_model.cpu(),
                target_patterns=args.target_patterns,
            )

            update_records.append({
                "round": round_id,
                "client_id": client_id,
                "modality": modality,
                "dominant_label": dominant_label,
                "update": update_vec,
            })

            local_models.append(local_model.to(args.device))
            client_sizes.append(len(client_data[client_id]))

        global_model = fedavg(global_model, local_models, client_sizes)
        global_model.to(args.device)

        if round_id in args.analysis_rounds or round_id == args.rounds:
                        train_metrics = evaluate(
                global_model,
                train_data,
                tokenizer,
                args,
                mode="both",
                max_samples=args.max_train_eval_samples,
            )

        val_metrics = evaluate(
                global_model,
                val_data,
                tokenizer,
                args,
                mode="both",
                max_samples=args.max_val_eval_samples,
            )

        test_metrics = evaluate(
                global_model,
                test_data,
                tokenizer,
                args,
                mode="both",
                max_samples=args.max_test_eval_samples,
            )

        print(
                f"Round {round_id:03d} | "
                f"Train Acc: {train_metrics['acc']:.4f} | "
                f"Val Acc: {val_metrics['acc']:.4f} | "
                f"Test Acc: {test_metrics['acc']:.4f} | "
                f"Test Macro-F1: {test_metrics['macro_f1']:.4f}"
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

        final_train_metrics = evaluate(
        global_model,
        train_data,
        tokenizer,
        args,
        mode="both",
        max_samples=args.max_train_eval_samples,
    )

    final_val_metrics = evaluate(
        global_model,
        val_data,
        tokenizer,
        args,
        mode="both",
        max_samples=args.max_val_eval_samples,
    )

    final_test_metrics = evaluate(
        global_model,
        test_data,
        tokenizer,
        args,
        mode="both",
        max_samples=args.max_test_eval_samples,
    )

    structure_metrics, all_mat = compute_structure_metrics(
        update_records,
        seed=args.seed,
    )

    global_num_classes = args.num_classes
    random_chance_acc = 1.0 / global_num_classes

    summary = {
        "setting_name": args.setting_name,
        "association": args.association,
        "num_clients": args.num_clients,
        "samples_per_client": args.samples_per_client,
        "allow_overlap": args.allow_overlap,
        "rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "lr": args.lr,
        "seed": args.seed,
        "model": "ResNet18+DistilBERT",
        "freeze_image_backbone": args.freeze_image_backbone,
        "freeze_text_backbone": args.freeze_text_backbone,

        # task definition
        "task_type": "global_6class_classification",
        "global_num_classes": global_num_classes,
        "random_chance_acc": random_chance_acc,
        "label_space_split_by_modality": False,

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

        # global_acc is kept for compatibility with previous result tables.
        # Here it means validation accuracy on the shared global 6-class task.
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

    # Save round logs
    pd.DataFrame(round_logs).to_csv(
        os.path.join(args.out_dir, "round_logs.csv"),
        index=False,
    )

    save_json(
        summary,
        os.path.join(args.out_dir, "summary.json"),
    )

    # Save update metadata
    meta_records = []

    for r in update_records:
        meta_records.append({
            "round": r["round"],
            "client_id": r["client_id"],
            "modality": r["modality"],
            "dominant_label": r["dominant_label"],
            "dominant_label_name": id_to_label_name[r["dominant_label"]],
        })

    pd.DataFrame(meta_records).to_csv(
        os.path.join(args.out_dir, "update_metadata.csv"),
        index=False,
    )

    return summary, round_logs, all_mat