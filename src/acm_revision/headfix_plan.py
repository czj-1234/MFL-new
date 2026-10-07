from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from .orchestrator import load_yaml


FORMAL_RUNS = (
    ("shadow_train", 142, 200),
    ("shadow_train", 143, 200),
    ("shadow_val", 242, 100),
    ("target", 42, 200),
    ("target", 43, 200),
    ("target", 44, 200),
    ("target", 45, 200),
    ("target", 46, 200),
)

# Two jobs per physical GPU. All shadow populations stay on server A so only
# target result folders need to be copied back for joint post-processing.
QUEUE_ASSIGNMENT = {
    "serverA_gpu0": (("shadow_train", 142), ("target", 42)),
    "serverA_gpu1": (("shadow_train", 143), ("shadow_val", 242)),
    "serverB_gpu0": (("target", 43), ("target", 44)),
    "serverB_gpu1": (("target", 45), ("target", 46)),
}


@dataclass(frozen=True)
class Job:
    job_id: str
    config_path: str
    population: str
    seed: int


def _exact_filter(group: str, basis: str, rank: int, alpha: float) -> dict:
    return {
        "name": "contrast_filter",
        "group": group,
        "basis_path": basis,
        "rank": int(rank),
        "alpha": float(alpha),
    }


def _hash_filter(group: str, basis: str, rank: int, alpha: float, old_manifest: dict) -> dict:
    hs = old_manifest["implicit_hash_lift"]
    return {
        "name": "implicit_hash_contrast",
        "group": group,
        "basis_path": basis,
        "rank": int(rank),
        "alpha": float(alpha),
        "projection_dim": int(hs["projection_dim"]),
        "projection_seed": int(hs["projection_seed"]),
        "basis_interpretation": "implicit_original_space_basis_B_equals_HtU",
    }


def _defense_cfg(old_manifest: dict, head_manifest: dict) -> dict:
    if str(old_manifest.get("status")) != "LOCKED":
        raise ValueError("Existing Defense70 manifest must have status=LOCKED.")
    if str(head_manifest.get("status")) != "LOCKED":
        raise ValueError("Head-fix manifest must have status=LOCKED.")
    if old_manifest.get("provenance", {}).get("target_data_used_for_basis_or_tuning") is not False:
        raise ValueError("Old Defense70 basis is not certified target-free.")
    if head_manifest.get("provenance", {}).get("target_data_used_for_basis_or_tuning") is not False:
        raise ValueError("Head-fix basis is not certified target-free.")

    old_basis = str(old_manifest["basis_paths"]["contrast"])
    old_strong = old_manifest["selected"]["strong"]
    old_rank = int(old_strong["rank"])
    old_alpha = float(old_strong["alpha"])

    head_basis = str(head_manifest["basis"]["path"])
    head_rank = int(head_manifest["selected"]["rank"])
    head_alpha = float(head_manifest["selected"]["alpha"])

    return {
        "name": "layerwise_mixed_filter",
        "protocol_name": "minimal_signed_headfix_plus_existing_layerwise",
        "layers": [
            _exact_filter("classifier_head", head_basis, head_rank, head_alpha),
            _exact_filter("fusion", old_basis, old_rank, old_alpha),
            _exact_filter("missing_modality", old_basis, old_rank, old_alpha),
            _hash_filter("image_encoder", old_basis, old_rank, old_alpha, old_manifest),
            _hash_filter("text_encoder", old_basis, old_rank, old_alpha, old_manifest),
        ],
        "all_defense_parameters_frozen_before_formal_target_runs": True,
        "minimal_headfix": {
            "classifier_head_method": "signed_matched_label_composition",
            "head_rank": head_rank,
            "head_alpha": head_alpha,
            "other_groups_reuse_original_strong_bases": True,
        },
    }


def _formal_config(
    base: dict,
    old_manifest: dict,
    head_manifest: dict,
    population: str,
    seed: int,
    samples: int,
) -> dict:
    cfg = copy.deepcopy(base)
    cfg["seed"] = int(seed)

    fed = cfg["federated"]
    fed.update(
        {
            "num_clients": 12,
            "partition_mode": "fixed",
            "samples_per_client": int(samples),
            "rounds": 150,
            "participation_rate": 1.0,
            "min_participants": 12,
            "local_epochs": 1,
            "optimizer": "adamw",
            "lr": 0.000001,
            "weight_decay": 0.05,
            "batch_size": 16,
            "max_local_steps": None,
            "aggregation": "fedavg",
            "fedprox_mu": 0.0,
        }
    )

    cfg.setdefault("evaluation", {})["eval_every"] = 5
    cfg["evaluation"]["num_workers"] = 2

    exp = cfg["experiment"]
    exp.update(
        {
            "population": population,
            "setting_name": "modality_exclusive",
            "concentration": 0.7,
            "output_root": "results/acm_revision/headfix_r150",
            "headfix_protocol": "HeadFix-Minimal-v1",
            "target_used_for_defense_selection": False,
        }
    )

    cfg["defense"] = _defense_cfg(old_manifest, head_manifest)

    # Match the final Core72 observation protocol so the existing no-defense
    # reference is directly comparable: exact head every round, the same
    # milestone schedule, 16,384 coordinate sketches, and 2,048-d full-update
    # signed feature-hash projection.
    milestones = [1, 5, 10, 20, 30, 50, 75, 100, 125, 150]
    cfg["privacy_capture"] = {
        "every_round_exact_groups": [
            "classifier_bias",
            "classifier_weight",
            "classifier_head",
        ],
        "milestone_exact_groups": ["fusion", "missing_modality"],
        "milestone_sketch_groups": [
            "image_encoder",
            "text_encoder",
            "all_shared",
            "full_update",
        ],
        "full_group_projection_groups": ["full_update"],
        "full_group_projection_dim": 2048,
        "full_group_projection_seed": 20260804,
        "milestone_rounds": milestones,
        "sketch_dim": 16384,
        "sketch_seed": 20260803,
        "representation_scope_note": (
            "HeadFix uses the final Core72 capture protocol: classifier head exact every round; "
            "fusion/missing exact at milestones; encoder/all_shared deterministic coordinate sketches; "
            "full_update a 2048-d deterministic signed feature-hash projection using every transmitted coordinate."
        ),
    }

    cfg["update_capture"] = {
        "save_all_checkpoints": False,
        "checkpoint_rounds": [],
        "model_checkpoint_rounds": [],
        "storage_dtype": "float32",
    }
    cfg.setdefault("resume", {})
    cfg["resume"].setdefault("enabled", True)
    cfg["resume"].setdefault("checkpoint_every", 5)
    cfg["resume"].setdefault("keep_last", 2)
    return cfg


def generate(
    head_manifest_path: str | Path = "results/acm_revision/headfix_prep/locked_headfix.yaml",
    old_manifest_path: str | Path = "results/acm_revision/defense70_prep/locked_params.yaml",
    output_dir: str | Path = "configs/acm_revision/generated/headfix",
    base_config: str | Path = "configs/acm_revision/hateful_memes.yaml",
) -> dict:
    head_manifest_path = Path(head_manifest_path)
    old_manifest_path = Path(old_manifest_path)
    with head_manifest_path.open("r", encoding="utf-8") as f:
        head_manifest = yaml.safe_load(f)
    with old_manifest_path.open("r", encoding="utf-8") as f:
        old_manifest = yaml.safe_load(f)
    base = load_yaml(base_config)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for old in output_dir.glob("*.yaml"):
        old.unlink()
    for old in output_dir.glob("*.tsv"):
        old.unlink()

    jobs: list[Job] = []
    for idx, (population, seed, samples) in enumerate(FORMAL_RUNS):
        cfg = _formal_config(base, old_manifest, head_manifest, population, seed, samples)
        job_id = f"headfix_{idx:02d}__{population}__seed{seed}"
        cfg["experiment"]["job_id"] = job_id
        config_path = output_dir / f"{job_id}.yaml"
        with config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        jobs.append(Job(job_id, str(config_path), population, int(seed)))

    if len(jobs) != 8:
        raise AssertionError(f"Expected 8 formal runs, got {len(jobs)}.")

    jobs_path = output_dir / "jobs.tsv"
    with jobs_path.open("w", encoding="utf-8") as f:
        f.write("job_id\tconfig_path\tpopulation\tseed\n")
        for j in jobs:
            f.write(f"{j.job_id}\t{j.config_path}\t{j.population}\t{j.seed}\n")

    lookup = {(j.population, j.seed): j for j in jobs}
    queue_files = {}
    for queue_name, assignments in QUEUE_ASSIGNMENT.items():
        qpath = output_dir / f"{queue_name}.tsv"
        with qpath.open("w", encoding="utf-8") as f:
            f.write("job_id\tconfig_path\tpopulation\tseed\n")
            for key in assignments:
                j = lookup[key]
                f.write(f"{j.job_id}\t{j.config_path}\t{j.population}\t{j.seed}\n")
        queue_files[queue_name] = str(qpath)

    audit = {
        "status": "PASS",
        "protocol": "HeadFix-Minimal-v1",
        "n_formal_runs": len(jobs),
        "capture": {
            "milestones": [1, 5, 10, 20, 30, 50, 75, 100, 125, 150],
            "all_shared_sketch_dim": 16384,
            "full_update_projection_dim": 2048,
        },
        "populations": {
            "shadow_train": [142, 143],
            "shadow_val": [242],
            "target": [42, 43, 44, 45, 46],
        },
        "queue_files": queue_files,
        "headfix_selected": head_manifest["selected"],
        "jobs": [j.__dict__ for j in jobs],
    }
    (output_dir / "generation_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the minimal 8-run classifier-head repair matrix."
    )
    parser.add_argument(
        "--head-manifest",
        default="results/acm_revision/headfix_prep/locked_headfix.yaml",
    )
    parser.add_argument(
        "--old-manifest",
        default="results/acm_revision/defense70_prep/locked_params.yaml",
    )
    parser.add_argument(
        "--output-dir",
        default="configs/acm_revision/generated/headfix",
    )
    parser.add_argument(
        "--base-config",
        default="configs/acm_revision/hateful_memes.yaml",
    )
    args = parser.parse_args()

    print(
        json.dumps(
            generate(
                args.head_manifest,
                args.old_manifest,
                args.output_dir,
                args.base_config,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
