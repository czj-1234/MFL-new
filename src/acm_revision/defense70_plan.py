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
    ("shadow_val", 242, 100),
    ("target", 42, 200),
    ("target", 43, 200),
    ("target", 44, 200),
    ("target", 45, 200),
    ("target", 46, 200),
)


@dataclass(frozen=True)
class OperatingPoint:
    op_id: str
    family: str
    tier: str
    defense: dict


def _exact_contrast(group: str, basis: str, rank: int, alpha: float) -> dict:
    return {
        "name": "contrast_filter",
        "group": group,
        "basis_path": basis,
        "rank": int(rank),
        "alpha": float(alpha),
    }


def _hash_filter(name: str, group: str, basis: str, rank: int, alpha: float, manifest: dict) -> dict:
    hs = manifest["implicit_hash_lift"]
    return {
        "name": name,
        "group": group,
        "basis_path": basis,
        "rank": int(rank),
        "alpha": float(alpha),
        "projection_dim": int(hs["projection_dim"]),
        "projection_seed": int(hs["projection_seed"]),
        "basis_interpretation": "implicit_original_space_basis_B_equals_HtU",
    }


def operating_points(manifest: dict) -> list[OperatingPoint]:
    if str(manifest.get("status")) != "LOCKED":
        raise ValueError("Defense70 manifest must have status=LOCKED")
    provenance = manifest.get("provenance", {})
    if provenance.get("target_data_used_for_basis_or_tuning") is not False:
        raise ValueError("Refusing to generate formal jobs: target-data isolation is not certified")

    paths = manifest["basis_paths"]
    weak = manifest["selected"]["weak"]
    strong = manifest["selected"]["strong"]
    head = manifest["selected"]["head_ablation"]
    fullbasis = manifest["selected"]["single_full_update_basis"]
    clip_weak = manifest["selected"]["clip_gaussian_weak"]
    clip_strong = manifest["selected"]["clip_gaussian_strong"]

    def layerwise(point: dict) -> dict:
        rank, alpha = int(point["rank"]), float(point["alpha"])
        return {
            "name": "layerwise_mixed_filter",
            "protocol_name": "contrast_aware_system_wide_layerwise",
            "layers": [
                _exact_contrast("classifier_head", paths["contrast"], rank, alpha),
                _exact_contrast("fusion", paths["contrast"], rank, alpha),
                _exact_contrast("missing_modality", paths["contrast"], rank, alpha),
                _hash_filter("implicit_hash_contrast", "image_encoder", paths["contrast"], rank, alpha, manifest),
                _hash_filter("implicit_hash_contrast", "text_encoder", paths["contrast"], rank, alpha, manifest),
            ],
            "all_defense_parameters_frozen_before_formal_target_runs": True,
        }

    return [
        OperatingPoint(
            "proposed_head_ablation",
            "proposed",
            "ablation",
            _exact_contrast("classifier_head", paths["contrast"], int(head["rank"]), float(head["alpha"])),
        ),
        OperatingPoint("proposed_system_weak", "proposed", "weak", layerwise(weak)),
        OperatingPoint("proposed_system_strong", "proposed", "strong", layerwise(strong)),
        OperatingPoint(
            "proposed_single_fullbasis",
            "proposed",
            "full_update_basis_ablation",
            _hash_filter(
                "implicit_hash_contrast",
                "full_update",
                paths["contrast"],
                int(fullbasis["rank"]),
                float(fullbasis["alpha"]),
                manifest,
            ),
        ),
        OperatingPoint(
            "pca_weak",
            "rank_matched_pca",
            "weak",
            _hash_filter(
                "implicit_hash_pca",
                "full_update",
                paths["pca_full_update"],
                int(weak["rank"]),
                float(weak["alpha"]),
                manifest,
            ),
        ),
        OperatingPoint(
            "pca_strong",
            "rank_matched_pca",
            "strong",
            _hash_filter(
                "implicit_hash_pca",
                "full_update",
                paths["pca_full_update"],
                int(strong["rank"]),
                float(strong["alpha"]),
                manifest,
            ),
        ),
        OperatingPoint(
            "random_weak",
            "rank_matched_random_subspace",
            "weak",
            _hash_filter(
                "implicit_hash_random",
                "full_update",
                paths["random_full_update"],
                int(weak["rank"]),
                float(weak["alpha"]),
                manifest,
            ),
        ),
        OperatingPoint(
            "random_strong",
            "rank_matched_random_subspace",
            "strong",
            _hash_filter(
                "implicit_hash_random",
                "full_update",
                paths["random_full_update"],
                int(strong["rank"]),
                float(strong["alpha"]),
                manifest,
            ),
        ),
        OperatingPoint(
            "clipgauss_weak",
            "clip_gaussian",
            "weak",
            {
                "name": "streaming_clip_gaussian",
                "group": "full_update",
                "clip_norm": float(clip_weak["clip_norm"]),
                "noise_l2_ratio": float(clip_weak["noise_l2_ratio"]),
                "calibrated_on_shadow_val_only": True,
            },
        ),
        OperatingPoint(
            "clipgauss_strong",
            "clip_gaussian",
            "strong",
            {
                "name": "streaming_clip_gaussian",
                "group": "full_update",
                "clip_norm": float(clip_strong["clip_norm"]),
                "noise_l2_ratio": float(clip_strong["noise_l2_ratio"]),
                "calibrated_on_shadow_val_only": True,
            },
        ),
    ]


def _formal_config(base: dict, manifest: dict, op: OperatingPoint, population: str, seed: int, samples: int) -> dict:
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
            "output_root": f"results/acm_revision/defense70_r150/{op.op_id}",
            "defense70_operating_point": op.op_id,
            "defense70_family": op.family,
            "defense70_tier": op.tier,
            "defense70_locked_manifest": "results/acm_revision/defense70_prep/locked_params.yaml",
            "target_used_for_defense_selection": False,
        }
    )
    cfg["defense"] = copy.deepcopy(op.defense)
    milestones = [1, 10, 30, 50, 75, 100, 125, 150]
    cfg["privacy_capture"] = {
        "every_round_exact_groups": ["classifier_bias", "classifier_weight", "classifier_head"],
        "milestone_exact_groups": ["fusion", "missing_modality"],
        "milestone_sketch_groups": ["image_encoder", "text_encoder", "all_shared", "full_update"],
        "full_group_projection_groups": ["full_update"],
        "full_group_projection_dim": 8192,
        "full_group_projection_seed": 20260804,
        "milestone_rounds": milestones,
        "sketch_dim": 16384,
        "sketch_seed": 20260803,
        "representation_scope_note": (
            "Classifier head is captured exactly each round. Fusion/missing groups are exact at milestones. "
            "Encoder/all_shared use deterministic coordinate sketches. full_update uses an 8192-d signed feature-hash "
            "in which every transmitted coordinate contributes; it is a compressed full-model observation, not an exact flat vector."
        ),
    }
    cfg["update_capture"] = {
        "save_all_checkpoints": False,
        "checkpoint_rounds": [],
        "model_checkpoint_rounds": [],
        "storage_dtype": "float32",
    }
    return cfg


def generate(
    manifest_path: str | Path,
    output_dir: str | Path = "configs/acm_revision/generated/defense70",
    base_config: str | Path = "configs/acm_revision/hateful_memes.yaml",
) -> dict:
    manifest_path = Path(manifest_path)
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)
    base = load_yaml(base_config)
    ops = operating_points(manifest)
    if len(ops) != 10:
        raise AssertionError(f"Expected 10 operating points, got {len(ops)}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Never leave stale scientific configs mixed with a newly locked manifest.
    for old in output_dir.glob("*.yaml"):
        old.unlink()

    rows = []
    idx = 0
    for op in ops:
        for population, seed, samples in FORMAL_RUNS:
            cfg = _formal_config(base, manifest, op, population, seed, samples)
            job_id = f"d70_{idx:02d}__{op.op_id}__{population}__seed{seed}"
            cfg["experiment"]["job_id"] = job_id
            path = output_dir / f"{job_id}.yaml"
            with path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
            rows.append((job_id, str(path), op.op_id, population, seed))
            idx += 1

    if len(rows) != 70:
        raise AssertionError(f"Expected 70 formal jobs, got {len(rows)}")
    jobs_path = output_dir / "jobs.tsv"
    with jobs_path.open("w", encoding="utf-8") as f:
        f.write("job_id\tconfig_path\n")
        for job_id, path, _, _, _ in rows:
            f.write(f"{job_id}\t{path}\n")

    audit = {
        "status": "PASS",
        "protocol": manifest.get("protocol"),
        "locked_manifest": str(manifest_path),
        "target_data_used_for_basis_or_tuning": manifest["provenance"]["target_data_used_for_basis_or_tuning"],
        "n_operating_points": len(ops),
        "n_runs_per_operating_point": len(FORMAL_RUNS),
        "n_formal_runs": len(rows),
        "operating_points": [
            {"op_id": op.op_id, "family": op.family, "tier": op.tier, "defense": op.defense}
            for op in ops
        ],
        "formal_populations": [
            {"population": p, "seed": s, "samples_per_client": n} for p, s, n in FORMAL_RUNS
        ],
        "jobs": [
            {"job_id": j, "config_path": p, "operating_point": o, "population": pop, "seed": s}
            for j, p, o, pop, s in rows
        ],
    }
    with (output_dir / "generation_audit.json").open("w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the frozen Defense70 formal matrix")
    parser.add_argument("--manifest", default="results/acm_revision/defense70_prep/locked_params.yaml")
    parser.add_argument("--output-dir", default="configs/acm_revision/generated/defense70")
    parser.add_argument("--base-config", default="configs/acm_revision/hateful_memes.yaml")
    args = parser.parse_args()
    print(json.dumps(generate(args.manifest, args.output_dir, args.base_config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
