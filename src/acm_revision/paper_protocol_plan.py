from __future__ import annotations

"""Generate canonical paper-protocol FL configs without changing historical runs.

Core72 requires committed Core72 config templates. Defense70/Complete ASLWF
additionally require the *actual* locked experiment manifests. The generated
files are instructions for independent future runs, NOT evidence that the
manuscript's historical numerical tables have been reproduced.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import yaml

from . import core72_plan, defense70_plan, headfix_plan
from .orchestrator import load_yaml

PAPER_MILESTONES = (1, 10, 30, 50, 75, 100, 125, 150)
PAPER_FULL_UPDATE_DIM = 2048
PAPER_ENCODER_SKETCH_DIM = 16384
REFERENCE_SEED = 20260804
SKETCH_SEED = 20260803


def _load_locked(path: Path, name: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing real {name} manifest: {path}")
    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("status") != "LOCKED":
        raise ValueError(f"{name} manifest is not LOCKED: {path}")
    if manifest.get("provenance", {}).get("target_data_used_for_basis_or_tuning") is not False:
        raise ValueError(f"{name} manifest lacks certified target-free provenance")
    return manifest


def _capture_paper(cfg: dict) -> None:
    cfg["privacy_capture"] = {
        "every_round_exact_groups": ["classifier_bias", "classifier_weight", "classifier_head"],
        "milestone_exact_groups": ["fusion", "missing_modality"],
        "milestone_sketch_groups": ["image_encoder", "text_encoder", "all_shared", "full_update"],
        "full_group_projection_groups": ["full_update"],
        "full_group_projection_dim": PAPER_FULL_UPDATE_DIM,
        "full_group_projection_seed": REFERENCE_SEED,
        "milestone_rounds": list(PAPER_MILESTONES),
        "sketch_dim": PAPER_ENCODER_SKETCH_DIM,
        "sketch_seed": SKETCH_SEED,
        "representation_scope_note": (
            "Paper-protocol exact classifier head every round; exact fusion and missing "
            "modality at eight milestones; 16384-coordinate sketches for image/text/all_shared; "
            "2048-dimensional signed all-coordinate full-update hash at eight milestones."
        ),
    }


def _finalize(cfg: dict, phase: str, job_id: str, result_root: Path) -> dict:
    cfg = copy.deepcopy(cfg)
    _capture_paper(cfg)
    exp = cfg.setdefault("experiment", {})
    exp["paper_repro_protocol"] = "ASLWF-manuscript-2026-10-17"
    exp["paper_repro_phase"] = phase
    exp["paper_repro_historical_result"] = False
    exp["job_id"] = job_id
    exp["output_root"] = str(result_root / phase / job_id)
    if cfg.get("federated", {}).get("rounds") != 150:
        raise AssertionError(f"Unexpected number of rounds in {job_id}")
    return cfg


def _write_jobs(configs: list[tuple[str, dict]], phase: str, out: Path, result_root: Path) -> dict:
    directory = out / phase
    directory.mkdir(parents=True, exist_ok=True)
    if list(directory.glob("*.yaml")):
        raise FileExistsError(f"Refusing to overwrite existing paper-protocol YAML in {directory}")
    jobs = []
    for job_id, config in configs:
        p = directory / (job_id + ".yaml")
        obj = _finalize(config, phase, job_id, result_root)
        p.write_text(yaml.safe_dump(obj, sort_keys=False, allow_unicode=True), encoding="utf-8")
        jobs.append({"id": job_id, "config": str(p),
                     "population": obj["experiment"]["population"],
                     "seed": int(obj["seed"])})
    (directory / "jobs.tsv").write_text(
        "job_id\tconfig_path\tpopulation\tseed\n" +
        "".join(f"{row['id']}\t{row['config']}\t{row['population']}\t{row['seed']}\n" for row in jobs),
        encoding="utf-8",
    )
    return {"phase": phase, "jobs": jobs, "count": len(jobs)}


def _core_templates(base_config: Path) -> list[tuple[str, dict]]:
    """Build all 72 paper jobs from the one canonical Hateful Memes YAML."""
    base = load_yaml(base_config)
    records = []
    for job in core72_plan.all_jobs():
        cfg = copy.deepcopy(base)
        cfg["seed"] = int(job.seed)
        cfg["federated"].update({
            "num_clients": 12, "partition_mode": "fixed",
            "samples_per_client": int(job.samples_per_client),
            "rounds": 150, "participation_rate": 1.0,
            "min_participants": 12, "local_epochs": 1,
            "optimizer": "adamw", "lr": 0.000001, "weight_decay": 0.05,
            "batch_size": 16, "max_local_steps": None,
            "aggregation": "fedavg", "fedprox_mu": 0.0,
        })
        cfg.setdefault("evaluation", {}).update({"eval_every": 5, "num_workers": 2})
        cfg["experiment"].update({
            "population": job.population, "setting_name": job.setting,
            "concentration": float(job.concentration),
            "balanced_attribution_control": job.concentration == 0.5,
        })
        cfg["defense"] = {"name": "none"}
        cfg["update_capture"] = {
            "save_all_checkpoints": False, "checkpoint_rounds": [],
            "model_checkpoint_rounds": [], "storage_dtype": "float32",
        }
        records.append((job.job_id, cfg))
    if len(records) != 72:
        raise AssertionError("Exactly 72 core FL jobs required")
    return records

def _defense_configs(base_config: Path, manifest: dict) -> list[tuple[str, dict]]:
    base = load_yaml(base_config)
    records = []
    for op in defense70_plan.operating_points(manifest):
        for population, seed, samples in defense70_plan.FORMAL_RUNS:
            job_id = f"{op.op_id}__{population}__seed{seed}"
            cfg = defense70_plan._formal_config(base, manifest, op, population, seed, samples)
            records.append((job_id, cfg))
    if len(records) != 70:
        raise AssertionError("Exactly 70 defense-baseline jobs are required")
    return records


def _complete_configs(base_config: Path, matched: dict, head: dict) -> list[tuple[str, dict]]:
    base = load_yaml(base_config)
    records = []
    for population, seed, samples in headfix_plan.FORMAL_RUNS:
        job_id = f"complete_aslwf__{population}__seed{seed}"
        cfg = headfix_plan._formal_config(base, matched, head, population, seed, samples)
        records.append((job_id, cfg))
    if len(records) != 8:
        raise AssertionError("Exactly eight complete ASLWF jobs are required")
    return records


def generate(
    stage: str = "core",
    root: Path = Path("."),
    output: Path = Path("configs/acm_revision/generated/paper_repro_20261017"),
) -> dict:
    root = root.resolve()
    out = (root / output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    result_root = Path("results/acm_revision/paper_repro_20261017")
    requested = ("core", "baselines", "complete") if stage == "all" else (stage,)
    report = {"protocol": "ASLWF-2026-10-17",
              "status": "PLANNED_NOT_RUN",
              "milestones": list(PAPER_MILESTONES),
              "full_update_dim": PAPER_FULL_UPDATE_DIM,
              "root": str(root), "phases": [],
              "historical_outputs_untouched": True}

    matched_path = root / "results/acm_revision/defense70_prep/locked_params.yaml"
    head_path = root / "results/acm_revision/headfix_prep/locked_headfix.yaml"
    base_config = root / "configs/acm_revision/hateful_memes.yaml"
    # Fail before any written job configs if a required original locked basis is absent.
    matched = _load_locked(matched_path, "matched") if any(x in requested for x in ("baselines", "complete")) else None
    head = _load_locked(head_path, "head") if "complete" in requested else None

    for name in requested:
        if name == "core":
            items = _core_templates(base_config)
        elif name == "baselines":
            items = _defense_configs(base_config, matched)
        elif name == "complete":
            items = _complete_configs(base_config, matched, head)
        else:
            raise ValueError(name)
        report["phases"].append(_write_jobs(items, name, out, result_root))

    manifest_hashes = {}
    for key, p in (("matched", matched_path), ("head", head_path)):
        if p.exists():
            manifest_hashes[key] = hashlib.sha256(p.read_bytes()).hexdigest()
    report["locked_manifest_sha256"] = manifest_hashes
    audit_path = out / "plan.json"
    audit_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("core", "baselines", "complete", "all"), default="core")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("configs/acm_revision/generated/paper_repro_20261017"))
    args = parser.parse_args()
    report = generate(args.stage, args.root, args.output)
    print(json.dumps({"status": report["status"],
                      "counts": {p["phase"]: p["count"] for p in report["phases"]},
                      "milestones": report["milestones"],
                      "full_update_dim": report["full_update_dim"]}, indent=2))


if __name__ == "__main__":
    main()
