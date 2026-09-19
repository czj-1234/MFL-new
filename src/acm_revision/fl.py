from __future__ import annotations

import copy
import json
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoTokenizer, CLIPImageProcessor

from .architectures import build_architecture
from .data_protocol import ClientSpec, load_json, make_client_specs, partition_clients_strict
from .defenses import LAYER_GROUPS, apply_defense_to_state, flatten_delta, state_delta


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _resolve_image_path(image_path: Optional[str]) -> Optional[Path]:
    if not image_path:
        return None
    image_path = str(image_path).replace("\\", "/")
    p = Path(image_path)
    if p.exists():
        return p
    candidates = [
        Path.cwd() / image_path,
        Path.cwd() / "data" / image_path,
        Path.cwd() / "data" / "raw" / image_path,
        Path.cwd() / "data" / "processed" / image_path,
        Path.cwd() / "data" / "images" / image_path,
        Path.cwd() / "data" / "MVSA" / image_path,
        Path.cwd() / "data" / "hateful_memes" / image_path,
        Path.cwd() / "data" / "Hateful Meme" / "hateful_memes" / image_path,
    ]
    return next((candidate for candidate in candidates if candidate.exists()), p)


def _image_path(item: Mapping) -> Optional[str]:
    for key in ("image", "image_path", "img", "path", "image_file", "filename"):
        if item.get(key):
            return str(item[key])
    return None


def _text(item: Mapping) -> str:
    for key in ("text", "sentence", "caption", "content"):
        if item.get(key) is not None:
            return str(item[key])
    return ""


def resolve_label(item: Mapping, mode: str, label_source: str = "auto") -> int:
    if label_source == "label":
        return int(item["label"])
    if label_source == "text":
        return int(item["text_label"])
    if label_source == "image":
        return int(item["image_label"])
    if "label" in item:
        return int(item["label"])
    if mode == "image" and "image_label" in item:
        return int(item["image_label"])
    if mode == "text" and "text_label" in item:
        return int(item["text_label"])
    if "text_label" in item:
        return int(item["text_label"])
    if "image_label" in item:
        return int(item["image_label"])
    raise KeyError("No usable label field found.")


class _TorchvisionProcessor:
    def __init__(self):
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __call__(self, images: Image.Image, return_tensors: str = "pt") -> dict:
        tensor = self.transform(images).unsqueeze(0)
        return {"pixel_values": tensor}


class RevisionDataset(Dataset):
    def __init__(
        self,
        data: Sequence[dict],
        tokenizer,
        image_processor,
        mode: str,
        max_text_len: int,
        label_source: str = "auto",
    ):
        self.data = list(data)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.mode = mode
        self.max_text_len = int(max_text_len)
        self.label_source = label_source

    def __len__(self) -> int:
        return len(self.data)

    def _load_image(self, item: Mapping) -> torch.Tensor:
        if self.mode not in ("image", "both"):
            return torch.zeros(3, 224, 224, dtype=torch.float32)
        try:
            path = _resolve_image_path(_image_path(item))
            if path is None:
                raise FileNotFoundError("Missing image path")
            with Image.open(path) as im:
                image = im.convert("RGB")
                return self.image_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        except Exception:
            return torch.zeros(3, 224, 224, dtype=torch.float32)

    def _tokenize(self, item: Mapping) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mode not in ("text", "both"):
            return (
                torch.zeros(self.max_text_len, dtype=torch.long),
                torch.zeros(self.max_text_len, dtype=torch.long),
            )
        encoded = self.tokenizer(
            _text(item),
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        )
        return encoded["input_ids"].squeeze(0), encoded["attention_mask"].squeeze(0)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        input_ids, attention_mask = self._tokenize(item)
        image = self._load_image(item)
        return {
            "pixel_values": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(resolve_label(item, self.mode, self.label_source), dtype=torch.long),
        }


def build_runtime(cfg: dict):
    model_cfg = cfg["model"]
    architecture = model_cfg.get("architecture", "clip_dual")
    tokenizer_name = model_cfg.get("tokenizer_name") or model_cfg.get("text_model_name") or model_cfg.get("image_model_name")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if architecture == "clip_dual":
        image_processor = CLIPImageProcessor.from_pretrained(model_cfg["image_model_name"])
    elif architecture == "resnet_roberta":
        image_processor = _TorchvisionProcessor()
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    model = build_architecture(cfg)
    return model, tokenizer, image_processor


def make_loader(
    data: Sequence[dict],
    tokenizer,
    image_processor,
    mode: str,
    cfg: dict,
    shuffle: bool,
    seed: int,
    label_source: str = "auto",
    batch_size: Optional[int] = None,
) -> DataLoader:
    dataset = RevisionDataset(
        data,
        tokenizer=tokenizer,
        image_processor=image_processor,
        mode=mode,
        max_text_len=int(cfg.get("evaluation", {}).get("max_text_len", 77)),
        label_source=label_source,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=int(batch_size or cfg["federated"]["batch_size"]),
        shuffle=shuffle,
        num_workers=int(cfg.get("evaluation", {}).get("num_workers", 4)),
        generator=generator if shuffle else None,
        pin_memory=torch.cuda.is_available(),
    )


def _optimizer(model: torch.nn.Module, cfg: dict):
    fed = cfg["federated"]
    params = [p for p in model.parameters() if p.requires_grad]
    name = str(fed.get("optimizer", "adamw")).lower()
    lr = float(fed["lr"])
    wd = float(fed.get("weight_decay", 0.0))
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=wd, momentum=float(fed.get("momentum", 0.9)))
    raise ValueError(f"Unsupported optimizer: {name}")


def local_train(
    global_model: torch.nn.Module,
    samples: Sequence[dict],
    tokenizer,
    image_processor,
    modality: str,
    cfg: dict,
    device: torch.device,
    seed: int,
) -> Tuple[Dict[str, torch.Tensor], dict]:
    local_model = copy.deepcopy(global_model).to(device)
    local_model.train()
    before = {k: v.detach().cpu().clone() for k, v in local_model.state_dict().items()}
    loader = make_loader(samples, tokenizer, image_processor, modality, cfg, True, seed, label_source="label")
    optimizer = _optimizer(local_model, cfg)
    local_epochs = int(cfg["federated"].get("local_epochs", 1))
    max_steps = cfg["federated"].get("max_local_steps")
    max_steps = int(max_steps) if max_steps is not None else None
    fedprox_mu = float(cfg["federated"].get("fedprox_mu", 0.0))

    total_loss = 0.0
    total = 0
    correct = 0
    steps = 0
    for _ in range(local_epochs):
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            logits = local_model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                setting=modality,
            )
            loss = F.cross_entropy(logits, labels)
            if fedprox_mu > 0:
                prox = torch.zeros((), device=device)
                for name, param in local_model.named_parameters():
                    if param.requires_grad and name in before:
                        prox = prox + torch.sum((param - before[name].to(device)) ** 2)
                loss = loss + 0.5 * fedprox_mu * prox
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            preds = logits.argmax(dim=1)
            total_loss += float(loss.item()) * labels.size(0)
            total += labels.size(0)
            correct += int((preds == labels).sum().item())
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break
        if max_steps is not None and steps >= max_steps:
            break

    after = {k: v.detach().cpu().clone() for k, v in local_model.state_dict().items()}
    del local_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return after, {
        "local_loss": total_loss / max(1, total),
        "local_acc": correct / max(1, total),
        "local_steps": steps,
    }


def weighted_average_states(states: Sequence[Mapping[str, torch.Tensor]], weights: Sequence[int]) -> Dict[str, torch.Tensor]:
    if not states:
        raise ValueError("No local states to aggregate.")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("Aggregation weight sum must be positive.")
    result: Dict[str, torch.Tensor] = {}
    for name in states[0]:
        ref = states[0][name]
        if not torch.is_floating_point(ref):
            result[name] = ref.detach().cpu().clone()
            continue
        acc = torch.zeros_like(ref, dtype=torch.float32, device="cpu")
        for state, weight in zip(states, weights):
            acc.add_(state[name].detach().cpu().float(), alpha=float(weight) / total)
        result[name] = acc.to(dtype=ref.dtype)
    return result


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data: Sequence[dict],
    tokenizer,
    image_processor,
    mode: str,
    cfg: dict,
    device: torch.device,
    label_source: str = "auto",
) -> dict:
    model = model.to(device)
    model.eval()
    loader = make_loader(
        data,
        tokenizer,
        image_processor,
        mode,
        cfg,
        shuffle=False,
        seed=int(cfg["seed"]),
        label_source=label_source,
        batch_size=int(cfg.get("evaluation", {}).get("eval_batch_size", 64)),
    )
    labels_all: List[int] = []
    preds_all: List[int] = []
    probs_all: List[np.ndarray] = []
    total_loss = 0.0
    total = 0
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        logits = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, setting=mode)
        loss = F.cross_entropy(logits, labels)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        total_loss += float(loss.item()) * labels.size(0)
        total += labels.size(0)
        labels_all.extend(labels.cpu().tolist())
        preds_all.extend(preds.cpu().tolist())
        probs_all.extend(probs.cpu().numpy())

    metrics = {
        "loss": total_loss / max(1, total),
        "acc": accuracy_score(labels_all, preds_all) if labels_all else 0.0,
        "macro_f1": f1_score(labels_all, preds_all, average="macro", zero_division=0) if labels_all else 0.0,
        "macro_precision": precision_score(labels_all, preds_all, average="macro", zero_division=0) if labels_all else 0.0,
        "macro_recall": recall_score(labels_all, preds_all, average="macro", zero_division=0) if labels_all else 0.0,
        "balanced_acc": balanced_accuracy_score(labels_all, preds_all) if labels_all else 0.0,
    }
    try:
        probs_np = np.asarray(probs_all)
        if int(cfg["data"]["num_classes"]) == 2:
            metrics["auroc"] = roc_auc_score(labels_all, probs_np[:, 1])
        else:
            metrics["auroc"] = roc_auc_score(labels_all, probs_np, multi_class="ovr", average="macro")
    except Exception:
        metrics["auroc"] = float("nan")
    model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {k: float(v) for k, v in metrics.items()}


def eval_mode(setting: str) -> str:
    if setting == "image_only":
        return "image"
    if setting == "text_only":
        return "text"
    return "both"


def eval_label_source(setting: str, cfg: dict) -> str:
    explicit = cfg.get("data", {}).get("eval_label_source")
    if explicit:
        return explicit
    if setting == "image_only":
        return "image"
    if setting == "text_only":
        return "text"
    return "auto"


def _save_update_npz(
    path: Path,
    raw_delta: Mapping[str, torch.Tensor],
    observed_delta: Mapping[str, torch.Tensor],
    groups: Sequence[str],
    storage_dtype: str,
) -> dict:
    dtype = np.float16 if storage_dtype == "float16" else np.float32
    arrays = {}
    dimensions = {}
    for group in groups:
        raw, _ = flatten_delta(raw_delta, group)
        obs, _ = flatten_delta(observed_delta, group)
        arrays[f"raw__{group}"] = raw.astype(dtype)
        arrays[f"observed__{group}"] = obs.astype(dtype)
        dimensions[group] = int(obs.size)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return dimensions


def _run_id(cfg: dict, population: str) -> str:
    exp = cfg["experiment"]
    model = cfg["model"].get("architecture", "clip_dual")
    fed = cfg["federated"]
    defense = cfg.get("defense", {}).get("name", "none")
    concentration = exp.get("concentration", exp.get("association", "iid"))
    return (
        f"{cfg['data'].get('name','dataset')}__{model}__{population}__{exp['setting_name']}__c{concentration}"
        f"__n{fed['num_clients']}__p{fed.get('participation_rate',1.0)}__{fed.get('aggregation','fedavg')}"
        f"__seed{cfg['seed']}__def-{defense}"
    ).replace("/", "-")


def run_strict_fl(cfg: dict, population: str = "target") -> dict:
    """Run one independent FL experiment under the strict revision protocol."""
    seed = int(cfg["seed"])
    set_global_seed(seed)
    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    run_id = _run_id(cfg, population)
    output_root = Path(cfg["experiment"].get("output_root", "results/acm_revision"))
    out_dir = output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "resolved_config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    pools_dir = Path(cfg["data"]["pools_dir"])
    population_file = {
        "shadow_train": "shadow_train.json",
        "shadow_val": "shadow_val.json",
        "target": "target.json",
    }[population]
    train_pool = load_json(pools_dir / population_file)
    task_val = load_json(pools_dir / "task_val.json")
    task_test = load_json(pools_dir / "task_test.json")

    num_classes = int(cfg["data"]["num_classes"])
    setting = cfg["experiment"]["setting_name"]
    concentration_value = cfg["experiment"].get("concentration", cfg["experiment"].get("association", "iid"))
    concentration = 1.0 / num_classes if str(concentration_value).lower() == "iid" else float(concentration_value)
    specs = make_client_specs(int(cfg["federated"]["num_clients"]), setting, num_classes)
    client_data, partition_manifest = partition_clients_strict(
        pool=train_pool,
        specs=specs,
        num_classes=num_classes,
        concentration=concentration,
        samples_per_client=cfg["federated"].get("samples_per_client"),
        seed=seed,
        partition_mode=cfg["federated"].get("partition_mode", "fixed"),
    )
    with (out_dir / "client_partition_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(partition_manifest, f, ensure_ascii=False, indent=2)

    global_model, tokenizer, image_processor = build_runtime(cfg)
    global_model.to("cpu")
    rounds = int(cfg["federated"]["rounds"])
    participation_rate = float(cfg["federated"].get("participation_rate", 1.0))
    min_participants = int(cfg["federated"].get("min_participants", 1))
    eval_every = int(cfg.get("evaluation", {}).get("eval_every", 5))
    save_all_checkpoints = bool(cfg.get("update_capture", {}).get("save_all_checkpoints", True))
    save_groups = cfg.get("update_capture", {}).get("groups", list(LAYER_GROUPS))
    storage_dtype = cfg.get("update_capture", {}).get("storage_dtype", "float16")
    defense_cfg = cfg.get("defense", {"name": "none"})

    metadata_rows: List[dict] = []
    eval_rows: List[dict] = []
    best_val = -float("inf")
    best_row = None
    rng = random.Random(seed + 7001)
    started = time.time()

    for round_id in range(1, rounds + 1):
        n_participants = max(min_participants, int(round(len(specs) * participation_rate)))
        n_participants = min(n_participants, len(specs))
        participants = sorted(rng.sample(specs, n_participants), key=lambda x: x.client_id)
        before_global = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
        local_states = []
        local_weights = []

        for spec in participants:
            local_seed = seed * 1_000_000 + round_id * 10_000 + spec.client_id
            after_raw, local_metrics = local_train(
                global_model,
                client_data[spec.client_id],
                tokenizer,
                image_processor,
                spec.modality,
                cfg,
                device,
                local_seed,
            )
            defended_state, defense_meta = apply_defense_to_state(before_global, after_raw, defense_cfg, seed=local_seed + 13)
            raw_delta = state_delta(before_global, after_raw)
            observed_delta = state_delta(before_global, defended_state)
            update_path = out_dir / "updates" / f"round_{round_id:04d}_client_{spec.client_id:04d}.npz"
            dimensions = _save_update_npz(update_path, raw_delta, observed_delta, save_groups, storage_dtype)

            values = client_data[spec.client_id]
            counts = Counter(int(x["label"]) for x in values)
            row = {
                "run_id": run_id,
                "population": population,
                "seed": seed,
                "round": round_id,
                "client_id": spec.client_id,
                "client_key": f"{run_id}::client{spec.client_id}",
                "setting_name": setting,
                "concentration": concentration,
                "modality": spec.modality,
                "dominant_label": spec.dominant_label,
                "num_samples": len(values),
                "participation_rate": participation_rate,
                "num_participants": n_participants,
                "architecture": cfg["model"].get("architecture", "clip_dual"),
                "aggregation": cfg["federated"].get("aggregation", "fedavg"),
                "optimizer": cfg["federated"].get("optimizer", "adamw"),
                "local_epochs": cfg["federated"].get("local_epochs", 1),
                "batch_size": cfg["federated"].get("batch_size"),
                "update_path": str(update_path),
                "local_loss": local_metrics["local_loss"],
                "local_acc": local_metrics["local_acc"],
                "defense_name": defense_meta.get("name", "none"),
                "defense_group": defense_meta.get("group"),
                "defense_before_norm": defense_meta.get("before_norm"),
                "defense_after_norm": defense_meta.get("after_norm"),
            }
            for label in range(num_classes):
                row[f"label_count_{label}"] = counts.get(label, 0)
                row[f"label_prop_{label}"] = counts.get(label, 0) / max(1, len(values))
            for group, dim in dimensions.items():
                row[f"dim_{group}"] = dim
            metadata_rows.append(row)
            local_states.append(defended_state)
            local_weights.append(len(values))

        global_state = weighted_average_states(local_states, local_weights)
        global_model.load_state_dict(global_state, strict=True)

        if save_all_checkpoints or round_id in set(cfg.get("update_capture", {}).get("checkpoint_rounds", [])):
            checkpoint_dir = out_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(global_model.state_dict(), checkpoint_dir / f"round_{round_id:04d}.pt")

        if round_id == 1 or round_id % eval_every == 0 or round_id == rounds:
            mode = eval_mode(setting)
            label_source = eval_label_source(setting, cfg)
            val_metrics = evaluate(global_model, task_val, tokenizer, image_processor, mode, cfg, device, label_source)
            test_metrics = evaluate(global_model, task_test, tokenizer, image_processor, mode, cfg, device, label_source)
            eval_row = {
                "run_id": run_id,
                "round": round_id,
                **{f"val_{k}": v for k, v in val_metrics.items()},
                **{f"test_{k}": v for k, v in test_metrics.items()},
            }
            eval_rows.append(eval_row)
            val_auroc = val_metrics.get("auroc", float("nan"))
            score = val_auroc if not np.isnan(val_auroc) else val_metrics.get("macro_f1", 0.0)
            if score > best_val:
                best_val = score
                best_row = dict(eval_row)
                torch.save(global_model.state_dict(), out_dir / "best_model.pt")

        pd.DataFrame(metadata_rows).to_csv(out_dir / "update_metadata.csv", index=False)
        pd.DataFrame(eval_rows).to_csv(out_dir / "round_metrics.csv", index=False)

    summary = {
        "run_id": run_id,
        "population": population,
        "seed": seed,
        "setting_name": setting,
        "concentration": concentration,
        "num_clients": len(specs),
        "participation_rate": participation_rate,
        "rounds": rounds,
        "architecture": cfg["model"].get("architecture", "clip_dual"),
        "aggregation": cfg["federated"].get("aggregation", "fedavg"),
        "optimizer": cfg["federated"].get("optimizer", "adamw"),
        "defense": defense_cfg,
        "runtime_seconds": time.time() - started,
        "best": best_row,
        "strict_data_overlap_allowed": False,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary
