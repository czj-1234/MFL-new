from __future__ import annotations

import copy
import errno
import hashlib
import json
import os
import random
import shutil
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

from . import fl as base
from .defenses import LAYER_GROUPS, apply_defense_to_state, state_delta


# privacy_capture_runner patches this symbol at runtime with the round-aware
# v2 capture saver, so defense runs keep the existing robust capture behavior.
_save_update_npz = base._save_update_npz


def _scientific_config_fingerprint(cfg: dict, population: str) -> str:
    """Fingerprint fields that must stay identical when resuming a run."""
    obj = copy.deepcopy(cfg)
    obj.pop("device", None)
    obj.pop("resume", None)
    exp = obj.get("experiment", {})
    exp.pop("output_root", None)
    exp.pop("job_id", None)
    obj["_resume_population"] = population
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _local_spool_dir() -> Path:
    base_dir = Path(os.environ.get("MFL_LOCAL_TMPDIR", tempfile.gettempdir()))
    spool = base_dir / f"mfl_resume_spool_{os.getuid()}"
    spool.mkdir(parents=True, exist_ok=True)
    return spool


def _copy_with_fsync(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, destination.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())


def _atomic_torch_save(obj, path: Path) -> None:
    """Save locally first, then atomically publish on the result filesystem."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    spool = _local_spool_dir()
    fd, tmp_name = tempfile.mkstemp(prefix="resume_", suffix=".pt", dir=spool)
    os.close(fd)
    local_tmp = Path(tmp_name)

    retryable = {
        getattr(errno, "ESTALE", 116),
        errno.EIO,
        errno.EAGAIN,
        getattr(errno, "ETIMEDOUT", 110),
    }
    attempts = int(os.environ.get("MFL_NFS_WRITE_RETRIES", "8"))

    try:
        torch.save(obj, local_tmp)
        if not local_tmp.exists() or local_tmp.stat().st_size <= 1024:
            raise OSError(errno.EIO, f"Invalid local resume checkpoint: {local_tmp}")

        last = None
        for attempt in range(1, attempts + 1):
            partial = path.with_name(f".{path.name}.partial.{os.getpid()}.{attempt}")
            try:
                partial.unlink(missing_ok=True)
                _copy_with_fsync(local_tmp, partial)
                os.replace(partial, path)
                if path.stat().st_size <= 1024:
                    raise OSError(errno.EIO, f"Published checkpoint is too small: {path}")
                return
            except OSError as exc:
                last = exc
                try:
                    partial.unlink(missing_ok=True)
                except OSError:
                    pass
                if exc.errno not in retryable or attempt >= attempts:
                    raise
                delay = min(30, 2 ** (attempt - 1))
                print(
                    f"[RESUME CHECKPOINT WRITE RETRY] {attempt}/{attempts} "
                    f"errno={exc.errno} sleep={delay}s path={path}",
                    flush=True,
                )
                time.sleep(delay)
        if last is not None:
            raise last
    finally:
        try:
            local_tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _atomic_dataframe_csv(frame: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.partial.{os.getpid()}")
    frame.to_csv(tmp, index=False)
    with tmp.open("rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _load_torch_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _capture_rng_state(participant_rng: random.Random) -> dict:
    state = {
        "python_global": random.getstate(),
        "numpy_global": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "participant_rng": participant_rng.getstate(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict, participant_rng: random.Random) -> None:
    if state.get("python_global") is not None:
        random.setstate(state["python_global"])
    if state.get("numpy_global") is not None:
        np.random.set_state(state["numpy_global"])
    if state.get("torch_cpu") is not None:
        torch.set_rng_state(state["torch_cpu"])
    if state.get("participant_rng") is not None:
        participant_rng.setstate(state["participant_rng"])
    if torch.cuda.is_available() and state.get("torch_cuda_all") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda_all"])


def _read_rows(path: Path, max_round: int) -> List[dict]:
    if not path.exists() or path.stat().st_size <= 0:
        return []
    frame = pd.read_csv(path)
    if "round" in frame.columns:
        frame = frame[frame["round"].astype(int) <= int(max_round)].copy()
    return frame.to_dict(orient="records")


def _clean_future_update_files(out_dir: Path, completed_round: int) -> None:
    updates = out_dir / "updates"
    if not updates.exists():
        return
    for p in updates.glob("round_*_client_*.npz"):
        try:
            rid = int(p.name.split("_")[1])
        except Exception:
            continue
        if rid > completed_round:
            p.unlink(missing_ok=True)


def _find_resume_checkpoint(resume_dir: Path, fingerprint: str, total_rounds: int):
    if not resume_dir.exists():
        return None, None
    for path in sorted(resume_dir.glob("round_*.pt"), reverse=True):
        try:
            cp = _load_torch_checkpoint(path)
            if cp.get("format") != "mfl_round_resume_v1":
                continue
            if cp.get("config_fingerprint") != fingerprint:
                raise RuntimeError(
                    f"Resume checkpoint config mismatch: {path}. "
                    "Do not resume an output directory with a different experiment config."
                )
            completed = int(cp.get("completed_round", -1))
            if 0 <= completed <= total_rounds:
                return path, cp
        except RuntimeError:
            raise
        except Exception as exc:
            print(
                f"[RESUME WARNING] Cannot load {path}: {exc}; trying older checkpoint.",
                flush=True,
            )
    return None, None


def _save_resume_checkpoint(
    *,
    resume_dir: Path,
    round_id: int,
    global_model: torch.nn.Module,
    best_val: float,
    best_row,
    participant_rng: random.Random,
    fingerprint: str,
    elapsed_seconds: float,
    keep_last: int,
) -> Path:
    resume_dir.mkdir(parents=True, exist_ok=True)
    path = resume_dir / f"round_{round_id:04d}.pt"
    payload = {
        "format": "mfl_round_resume_v1",
        "completed_round": int(round_id),
        "global_state": {
            k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()
        },
        "best_val": float(best_val),
        "best_row": best_row,
        "rng_state": _capture_rng_state(participant_rng),
        "config_fingerprint": fingerprint,
        "elapsed_seconds": float(elapsed_seconds),
        "saved_at_unix": float(time.time()),
    }
    _atomic_torch_save(payload, path)

    checkpoints = sorted(resume_dir.glob("round_*.pt"))
    while len(checkpoints) > max(1, int(keep_last)):
        old = checkpoints.pop(0)
        if old != path:
            old.unlink(missing_ok=True)
    return path


def run_strict_fl(cfg: dict, population: str = "target") -> dict:
    """Strict FL runner with true communication-round checkpoint/resume."""
    seed = int(cfg["seed"])
    base.set_global_seed(seed)
    device = torch.device(
        cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    run_id = base._run_id(cfg, population)
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
    train_pool = base.load_json(pools_dir / population_file)
    task_val = base.load_json(pools_dir / "task_val.json")
    task_test = base.load_json(pools_dir / "task_test.json")

    num_classes = int(cfg["data"]["num_classes"])
    setting = cfg["experiment"]["setting_name"]
    concentration_value = cfg["experiment"].get(
        "concentration", cfg["experiment"].get("association", "iid")
    )
    concentration = (
        1.0 / num_classes
        if str(concentration_value).lower() == "iid"
        else float(concentration_value)
    )
    specs = base.make_client_specs(
        int(cfg["federated"]["num_clients"]), setting, num_classes
    )
    client_data, partition_manifest = base.partition_clients_strict(
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

    global_model, tokenizer, image_processor = base.build_runtime(cfg)
    global_model.to("cpu")

    rounds = int(cfg["federated"]["rounds"])
    participation_rate = float(cfg["federated"].get("participation_rate", 1.0))
    min_participants = int(cfg["federated"].get("min_participants", 1))
    eval_every = int(cfg.get("evaluation", {}).get("eval_every", 5))
    save_all_checkpoints = bool(
        cfg.get("update_capture", {}).get("save_all_checkpoints", True)
    )
    save_groups = cfg.get("update_capture", {}).get("groups", list(LAYER_GROUPS))
    storage_dtype = cfg.get("update_capture", {}).get("storage_dtype", "float16")
    defense_cfg = cfg.get("defense", {"name": "none"})

    resume_cfg = cfg.get("resume", {})
    resume_enabled = bool(resume_cfg.get("enabled", True))
    checkpoint_every = int(resume_cfg.get("checkpoint_every", 5))
    keep_last = int(resume_cfg.get("keep_last", 2))
    if checkpoint_every <= 0:
        raise ValueError("resume.checkpoint_every must be positive")
    if keep_last <= 0:
        raise ValueError("resume.keep_last must be positive")

    fingerprint = _scientific_config_fingerprint(cfg, population)
    resume_dir = out_dir / "resume"

    metadata_rows: List[dict] = []
    eval_rows: List[dict] = []
    best_val = -float("inf")
    best_row = None
    participant_rng = random.Random(seed + 7001)
    completed_round = 0
    elapsed_before = 0.0

    if resume_enabled:
        cp_path, cp = _find_resume_checkpoint(
            resume_dir, fingerprint=fingerprint, total_rounds=rounds
        )
        if cp is not None:
            completed_round = int(cp["completed_round"])
            global_model.load_state_dict(cp["global_state"], strict=True)
            best_val = float(cp.get("best_val", -float("inf")))
            best_row = cp.get("best_row")
            elapsed_before = float(cp.get("elapsed_seconds", 0.0))
            metadata_rows = _read_rows(out_dir / "update_metadata.csv", completed_round)
            eval_rows = _read_rows(out_dir / "round_metrics.csv", completed_round)
            _clean_future_update_files(out_dir, completed_round)
            _atomic_dataframe_csv(pd.DataFrame(metadata_rows), out_dir / "update_metadata.csv")
            _atomic_dataframe_csv(pd.DataFrame(eval_rows), out_dir / "round_metrics.csv")
            _restore_rng_state(cp.get("rng_state", {}), participant_rng)
            print(
                f"[ROUND RESUME] run={run_id} checkpoint={cp_path.name} "
                f"completed_round={completed_round}; continuing at round {completed_round + 1}",
                flush=True,
            )
        else:
            print(
                f"[ROUND RESUME] no checkpoint found; starting run={run_id} from round 1",
                flush=True,
            )

    started = time.time()

    for round_id in range(completed_round + 1, rounds + 1):
        n_participants = max(
            min_participants, int(round(len(specs) * participation_rate))
        )
        n_participants = min(n_participants, len(specs))
        participants = sorted(
            participant_rng.sample(specs, n_participants), key=lambda x: x.client_id
        )
        before_global = {
            k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()
        }
        local_states = []
        local_weights = []

        for spec in participants:
            local_seed = seed * 1_000_000 + round_id * 10_000 + spec.client_id
            after_raw, local_metrics = base.local_train(
                global_model,
                client_data[spec.client_id],
                tokenizer,
                image_processor,
                spec.modality,
                cfg,
                device,
                local_seed,
            )
            defended_state, defense_meta = apply_defense_to_state(
                before_global, after_raw, defense_cfg, seed=local_seed + 13
            )
            raw_delta = state_delta(before_global, after_raw)
            observed_delta = state_delta(before_global, defended_state)
            update_path = (
                out_dir / "updates" / f"round_{round_id:04d}_client_{spec.client_id:04d}.npz"
            )
            dimensions = _save_update_npz(
                update_path, raw_delta, observed_delta, save_groups, storage_dtype
            )

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

        global_state = base.weighted_average_states(local_states, local_weights)
        global_model.load_state_dict(global_state, strict=True)

        fixed_checkpoint_rounds = set(
            cfg.get("update_capture", {}).get("checkpoint_rounds", [])
        )
        if save_all_checkpoints or round_id in fixed_checkpoint_rounds:
            checkpoint_dir = out_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                global_model.state_dict(), checkpoint_dir / f"round_{round_id:04d}.pt"
            )

        if round_id == 1 or round_id % eval_every == 0 or round_id == rounds:
            mode = base.eval_mode(setting)
            label_source = base.eval_label_source(setting, cfg)
            val_metrics = base.evaluate(
                global_model,
                task_val,
                tokenizer,
                image_processor,
                mode,
                cfg,
                device,
                label_source,
            )
            test_metrics = base.evaluate(
                global_model,
                task_test,
                tokenizer,
                image_processor,
                mode,
                cfg,
                device,
                label_source,
            )
            eval_row = {
                "run_id": run_id,
                "round": round_id,
                **{f"val_{k}": v for k, v in val_metrics.items()},
                **{f"test_{k}": v for k, v in test_metrics.items()},
            }
            eval_rows.append(eval_row)
            val_auroc = val_metrics.get("auroc", float("nan"))
            score = (
                val_auroc
                if not np.isnan(val_auroc)
                else val_metrics.get("macro_f1", 0.0)
            )
            if score > best_val:
                best_val = score
                best_row = dict(eval_row)
                torch.save(global_model.state_dict(), out_dir / "best_model.pt")

        _atomic_dataframe_csv(pd.DataFrame(metadata_rows), out_dir / "update_metadata.csv")
        _atomic_dataframe_csv(pd.DataFrame(eval_rows), out_dir / "round_metrics.csv")

        if resume_enabled and (round_id % checkpoint_every == 0 or round_id == rounds):
            elapsed_now = elapsed_before + (time.time() - started)
            cp_path = _save_resume_checkpoint(
                resume_dir=resume_dir,
                round_id=round_id,
                global_model=global_model,
                best_val=best_val,
                best_row=best_row,
                participant_rng=participant_rng,
                fingerprint=fingerprint,
                elapsed_seconds=elapsed_now,
                keep_last=keep_last,
            )
            print(
                f"[ROUND CHECKPOINT] completed_round={round_id} saved={cp_path}",
                flush=True,
            )

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
        "runtime_seconds": elapsed_before + (time.time() - started),
        "best": best_row,
        "strict_data_overlap_allowed": False,
        "round_resume": {
            "enabled": resume_enabled,
            "checkpoint_every": checkpoint_every,
            "keep_last": keep_last,
            "resumed_from_round": completed_round,
        },
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary
