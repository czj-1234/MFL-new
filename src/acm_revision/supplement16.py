"""Two locked ASLWF supplemental conditions (eight runs each).

c09: Transfer the frozen centered c=.7 ASLWF to c=.9 without re-tuning.
uncentered: Refit ONLY the four matched-contrast group bases without centering.
            Keep the attacker-guided classifier head and all hyperparameters.
Neither path accesses target data during basis construction or selection.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from . import defense70_prepare as prep
from . import headfix_plan
from .data_protocol import create_matched_composition_pair, load_json, make_client_specs
from .fl import build_runtime, local_train, set_global_seed
from .orchestrator import load_yaml

REPO = Path(".")
OLD_LOCK = Path("results/acm_revision/defense70_prep/locked_params.yaml")
HEAD_LOCK = Path("results/acm_revision/headfix_prep/locked_headfix.yaml")
REF_CONFIG = Path("configs/acm_revision/generated/defense70_prep/reference_seed142.yaml")
OUT = Path("results/acm_revision/final16_20261022")
CONFIG_OUT = Path("configs/acm_revision/generated/final16_20261022")
OTHER_GROUPS = ("fusion", "missing_modality", "image_encoder", "text_encoder")
MILESTONES = (1, 10, 30, 50, 75, 100, 125, 150)
EXPECTED = (("shadow_train", 142, 200), ("shadow_train", 143, 200),
            ("shadow_val", 242, 100),
            ("target", 42, 200), ("target", 43, 200),
            ("target", 44, 200), ("target", 45, 200), ("target", 46, 200))
BASIS = OUT / "preparation" / "uncentered_contrast_bases.npz"
BASIS_JSON = BASIS.with_suffix(".json")


def require_file(path):
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Required real artifact absent: {p}. No synthetic or fallback basis allowed.")
    return p


def sha(path):
    return hashlib.sha256(require_file(path).read_bytes()).hexdigest()


def load_locks():
    matched = yaml.safe_load(require_file(OLD_LOCK).read_text(encoding="utf-8"))
    head = yaml.safe_load(require_file(HEAD_LOCK).read_text(encoding="utf-8"))
    for name, obj in (("Defense70", matched), ("HeadFix", head)):
        if obj.get("status") != "LOCKED":
            raise ValueError(f"{name} manifest not LOCKED")
        if obj.get("provenance", {}).get("target_data_used_for_basis_or_tuning") is not False:
            raise ValueError(f"{name} manifest does not certify shadow/target isolation")
    for p in (*matched["basis_paths"].values(), head["basis"]["path"]):
        require_file(p)
    scope = matched["scientific_scope"]
    for key, value in (("setting", "modality_exclusive"), ("concentration", .7),
                       ("rounds", 150), ("num_clients", 12),
                       ("participation_rate", 1.0)):
        if scope.get(key) != value:
            raise ValueError(f"Existing frozen method mismatches {key}: {scope.get(key)}")
    return matched, head


def uncentered_lowrank(rows, rank):
    """Uncentered SVD via sample-space Gram; no subtraction of the mean."""
    matrix = np.stack(rows).astype(np.float32, copy=False)
    gram = matrix @ matrix.T
    eig, vectors = np.linalg.eigh(gram.astype(np.float64, copy=False))
    idx = np.argsort(eig)[::-1]
    eig, vectors = np.maximum(eig[idx], 0), vectors[:, idx]
    positive = eig > max(1e-12, (eig[0] * 1e-10 if len(eig) else 1e-12))
    actual = min(int(rank), int(positive.sum()), matrix.shape[1])
    if actual < rank:
        raise ValueError(f"Uncentered contrast has rank {actual}, requires {rank}")
    basis = (matrix.T @ vectors[:, :actual]) / (np.sqrt(eig[:actual])[None, :] + 1e-12)
    q, _ = np.linalg.qr(basis.astype(np.float64, copy=False))
    return q[:, :actual].astype(np.float32)


def prepare_uncentered():
    matched, head = load_locks()
    reference = Path(matched["provenance"]["reference_run_dir"])
    assert tuple(int(x) for x in matched["provenance"]["reference_rounds"]) == MILESTONES
    if int(matched["implicit_hash_lift"]["projection_dim"]) != prep.PROJECTION_DIM:
        raise ValueError("Frozen hash-lift dimension differs from original Defense70 extractor")
    if int(matched["implicit_hash_lift"]["projection_seed"]) != prep.PROJECTION_SEED:
        raise ValueError("Frozen hash-lift seed differs from original Defense70 extractor")
    checkpoints = [require_file(reference / "checkpoints" / f"round_{r:04d}.pt")
                   for r in MILESTONES]
    source_hashes = {"matched_lock": sha(OLD_LOCK), "head_lock": sha(HEAD_LOCK)}
    source_hashes["reference_config"] = sha(REF_CONFIG)
    source_hashes["centered_basis"] = sha(matched["basis_paths"]["contrast"])
    source_hashes["checkpoints"] = {str(i): sha(p) for i, p in zip(MILESTONES, checkpoints)}
    if BASIS.exists() or BASIS_JSON.exists():
        if not (BASIS.exists() and BASIS_JSON.exists()):
            raise ValueError("Incomplete prior basis preparation; remove/reconcile manually")
        metadata = json.loads(BASIS_JSON.read_text(encoding="utf-8"))
        if metadata.get("source_hashes") != source_hashes:
            raise ValueError("Existing basis provenance differs: refusing to overwrite/reuse")
        with np.load(BASIS, allow_pickle=False) as f:
            if set(f.files) != set(OTHER_GROUPS):
                raise ValueError("Existing basis has wrong groups")
        print("[REUSE] Verified uncentered basis and all input hashes", flush=True)
        return

    cfg = load_yaml(REF_CONFIG)
    if (cfg["experiment"].get("population") != "shadow_train"
            or float(cfg["experiment"].get("concentration")) != .7
            or int(cfg["seed"]) != 142):
        raise ValueError("Reference config must be shadow_train seed142, concentration .7")
    if int(cfg["federated"]["num_clients"]) != 12:
        raise ValueError("Reference requires 12 clients")
    if Path(cfg["experiment"]["output_root"]) / prep._run_id(cfg, "shadow_train") != reference:
        raise ValueError("Reference path does not match locked manifest")
    branch_size = int(matched["provenance"]["matched_contrast_samples_per_branch"])
    if branch_size < 1:
        raise ValueError("Invalid matched sample budget")

    pool = load_json(require_file(Path(cfg["data"]["pools_dir"]) / "shadow_train.json"))
    specs = make_client_specs(12, "modality_exclusive", int(cfg["data"]["num_classes"]))
    set_global_seed(142)
    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, tokenizer, processor = build_runtime(cfg)
    rows = {g: [] for g in OTHER_GROUPS}
    print("[FIT] shadow-only uncentered matched contrast, 8 checkpoints x 12 clients x 2 branches", flush=True)
    for round_id, ckpt in zip(MILESTONES, checkpoints):
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state, strict=True)
        model.to("cpu")
        before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        for spec in specs:
            pair_seed = 142 * 1_000_000 + round_id * 10_000 + spec.client_id
            concentrated, balanced, _ = create_matched_composition_pair(
                pool=pool, spec=spec, num_classes=int(cfg["data"]["num_classes"]),
                samples_per_client=branch_size, concentrated=.7, balanced=.5,
                seed=pair_seed)
            conc_state, _ = local_train(model, concentrated, tokenizer, processor,
                                        spec.modality, cfg, device, pair_seed)
            cf = prep._extract_features(before, conc_state)
            del conc_state
            bal_state, _ = local_train(model, balanced, tokenizer, processor,
                                       spec.modality, cfg, device, pair_seed)
            bf = prep._extract_features(before, bal_state)
            del bal_state
            for group in OTHER_GROUPS:
                rows[group].append((cf[group] - bf[group]).astype(np.float32))
            del cf, bf
            gc.collect()
            # Shadow-only basis preparation happens before the 8 formal jobs;
            # show its own live progress so the launcher never appears idle.
            print(
                f"[PREP PROGRESS] reference_round={round_id}/150 "
                f"matched_client={spec.client_id} "
                f"pairs_done_this_checkpoint={len(rows[OTHER_GROUPS[0]]) % 12 or 12}/12",
                flush=True,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del before, state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[FIT] checkpoint {round_id}/150 done", flush=True)

    old_basis_path = matched["basis_paths"]["contrast"]
    new_bases, parity = {}, {}
    with np.load(old_basis_path, allow_pickle=False) as existing:
        for group in OTHER_GROUPS:
            # Validate the re-created centered design before changing ONLY centering.
            centered, _ = prep._fit_lowrank(rows[group], prep.MAX_RANK)
            baseline = np.asarray(existing[group], dtype=np.float32)
            rank = min(centered.shape[1], baseline.shape[1])
            if rank < int(matched["selected"]["strong"]["rank"]):
                raise ValueError(f"Insufficient comparable rank for {group}")
            cosines = np.linalg.svd(
                np.linalg.qr(centered[:, :rank])[0].T @
                np.linalg.qr(baseline[:, :rank])[0],
                compute_uv=False)
            min_cosine = float(min(cosines))
            parity[group] = min_cosine
            if min_cosine < .995:
                raise ValueError(
                    f"Shadow contrast reconstruction differs from the original centered basis "
                    f"for {group}: min principal cosine={min_cosine:.6f}. "
                    "Do not silently run a confounded ablation.")
            new_bases[group] = uncentered_lowrank(rows[group], prep.MAX_RANK)
            print(f"[CHECK] centered parity {group}: {min_cosine:.6f}", flush=True)

    BASIS.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = BASIS.with_name("uncentered_contrast_bases.tmp.npz")
    np.savez_compressed(tmp_path, **new_bases)
    tmp_path.replace(BASIS)
    metadata = {"method": "uncentered_checkpoint_matched_contrast",
                "only_changed_factor": "mean_subtraction_in_matched_contrast_SVD",
                "head_basis_unchanged": True, "target_data_used": False,
                "reference_rounds": list(MILESTONES),
                "reference_concentration": .7, "balanced_control": .5,
                "samples_per_branch": branch_size,
                "defense_projection_dim": prep.PROJECTION_DIM,
                "capture_full_update_dim": 2048,
                "source_hashes": source_hashes, "centered_parity_min_cosine": parity,
                "basis_sha256": sha(BASIS)}
    BASIS_JSON.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[LOCKED] {BASIS}; centered parity confirmed", flush=True)


def validate_config(cfg, phase, matched, head):
    f = cfg["federated"]
    e = cfg["experiment"]
    d = cfg["defense"]
    assert f["num_clients"] == 12 and f["rounds"] == 150
    assert f["participation_rate"] == 1.0 and f["min_participants"] == 12
    assert (f["batch_size"], f["local_epochs"], f["lr"]) == (16, 1, .000001)
    assert cfg["privacy_capture"]["milestone_rounds"] == list(MILESTONES)
    assert cfg["privacy_capture"]["full_group_projection_dim"] == 2048
    assert cfg["privacy_capture"]["sketch_dim"] == 16384
    assert e["setting_name"] == "modality_exclusive"
    assert e["concentration"] == (.9 if phase == "c09" else .7)
    assert d["name"] == "layerwise_mixed_filter"
    assert [x["group"] for x in d["layers"]] == [
        "classifier_head", "fusion", "missing_modality", "image_encoder", "text_encoder"]
    frozen = headfix_plan._defense_cfg(matched, head)
    assert d["layers"][0] == frozen["layers"][0], "Attacker-refined head must not change"
    assert d["relative_l2_budget"] == frozen["relative_l2_budget"]
    for a, b in zip(d["layers"][1:], frozen["layers"][1:]):
        assert {k: v for k, v in a.items() if k != "basis_path"} == {
            k: v for k, v in b.items() if k != "basis_path"}
        if phase == "c09":
            assert a["basis_path"] == b["basis_path"]
        else:
            assert a["basis_path"] == str(BASIS)
    for layer in d["layers"]:
        require_file(layer["basis_path"])


def generate(phase):
    if phase not in ("uncentered", "c09"):
        raise ValueError(phase)
    matched, head = load_locks()
    if phase == "uncentered":
        require_file(BASIS)
        meta = json.loads(require_file(BASIS_JSON).read_text(encoding="utf-8"))
        if sha(BASIS) != meta["basis_sha256"]:
            raise ValueError("Uncentered basis altered after preparation")
        if sha(OLD_LOCK) != meta["source_hashes"]["matched_lock"]:
            raise ValueError("Frozen matched manifest changed after preparation")
        if sha(HEAD_LOCK) != meta["source_hashes"]["head_lock"]:
            raise ValueError("Frozen head manifest changed after preparation")
    base = load_yaml("configs/acm_revision/hateful_memes.yaml")
    dst = CONFIG_OUT / phase
    dst.mkdir(parents=True, exist_ok=True)
    output_root = OUT / phase
    configs = []
    for pop, seed, samples in EXPECTED:
        config = headfix_plan._formal_config(base, matched, head, pop, seed, samples)
        config["experiment"]["concentration"] = .9 if phase == "c09" else .7
        config["experiment"]["output_root"] = str(output_root)
        config["experiment"]["supplement16"] = {
            "phase": phase, "target_used_for_defense_selection": False,
            "frozen_from": "Complete ASLWF c=0.7",
            "only_changed_factor": "concentration" if phase == "c09" else "contrast_centering"}
        if phase == "uncentered":
            for layer in config["defense"]["layers"]:
                if layer["group"] in OTHER_GROUPS:
                    layer["basis_path"] = str(BASIS)
            config["defense"]["protocol_name"] = "uncentered_contrast_same_refined_head"
        validate_config(config, phase, matched, head)
        job_id = f"s16_{phase}_{pop}_seed{seed}"
        config["experiment"]["job_id"] = job_id
        path = dst / f"{job_id}.yaml"
        if path.exists() and yaml.safe_load(path.read_text(encoding="utf-8")) != config:
            raise ValueError(f"Existing config has different scientific specification: {path}")
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        configs.append((job_id, str(path), pop, seed))
    if len(configs) != 8:
        raise AssertionError("Exactly eight jobs required per condition")
    assignments = (configs[:1] + configs[3::2], configs[1:3] + configs[4::2])
    # No omitted/duplicated jobs: four jobs per GPU
    if len(assignments[0]) != 4 or len(assignments[1]) != 4:
        raise AssertionError("Expected four jobs per GPU")
    assert {j[0] for q in assignments for j in q} == {j[0] for j in configs}
    for gpu, queue in enumerate(assignments):
        queue_path = dst / f"gpu{gpu}.tsv"
        queue_content = "job_id\tconfig_path\tpopulation\tseed\n" + "".join(
            "\t".join(map(str, item)) + "\n" for item in queue)
        if queue_path.exists() and queue_path.read_text(encoding="utf-8") != queue_content:
            raise ValueError(f"Existing queue differs: {queue_path}")
        queue_path.write_text(queue_content, encoding="utf-8")
    report = {"status": "PLANNED_NOT_RUN", "phase": phase, "count": 8,
              "gpu0": [row[0] for row in assignments[0]],
              "gpu1": [row[0] for row in assignments[1]],
              "matched_manifest_sha": sha(OLD_LOCK), "head_manifest_sha": sha(HEAD_LOCK),
              "uncentered_basis_sha": sha(BASIS) if phase == "uncentered" else None,
              "capture_milestones": list(MILESTONES), "capture_full_update_dim": 2048,
              "unchanged_refined_head": True,
              "frozen_defense_selection": True}
    (dst / "plan.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


def verify(phase):
    matched, head = load_locks()
    dst = CONFIG_OUT / phase
    runs = []
    for pop, seed, _ in EXPECTED:
        p = dst / f"s16_{phase}_{pop}_seed{seed}.yaml"
        config = load_yaml(require_file(p))
        validate_config(config, phase, matched, head)
        f, e = config["federated"], config["experiment"]
        run_id = (f"{config['data'].get('name','dataset')}__{config['model'].get('architecture','clip_dual')}"
                  f"__{pop}__{e['setting_name']}__c{e['concentration']}__n{f['num_clients']}"
                  f"__p{f['participation_rate']}__{f.get('aggregation','fedavg')}"
                  f"__seed{seed}__def-{config['defense']['name']}")
        folder = Path(e["output_root"]) / run_id
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing run: {folder}")
        summary = json.loads(require_file(folder / "summary.json").read_text(encoding="utf-8"))
        manifest = json.loads(require_file(folder / "capture_manifest.json").read_text(encoding="utf-8"))
        if (manifest.get("status") != "PASS" or int(summary.get("rounds", -1)) != 150
                or summary.get("population") != pop or int(summary.get("seed", -1)) != seed
                or tuple(sorted(manifest.get("milestone_rounds_present", []))) != MILESTONES):
            raise RuntimeError(f"Incomplete/inconsistent result {folder}")
        runs.append((pop, seed, folder, summary))
    results = []
    for pop, seed, folder, summary in runs:
        if pop != "target":
            continue
        metadata = pd.read_csv(folder / "update_metadata.csv")
        if int(metadata["round"].max()) != 150:
            raise ValueError(f"Incomplete update metadata: {folder}")
        rec = {"seed": seed,
               "task_test_auroc": (summary.get("best") or {}).get("test_auroc")}
        for group in ("head", "full"):
            col = f"defense_l2_{group}_post"
            if col not in metadata or metadata[col].dropna().empty:
                raise ValueError(f"Missing exact actual original-space L2: {folder} {col}")
            vals = metadata[col].dropna().to_numpy(dtype=np.float64)
            rec[f"{group}_mean_actual_relative_l2"] = float(np.mean(vals))
            rec[f"{group}_median_actual_relative_l2"] = float(np.median(vals))
            rec[f"{group}_p95_actual_relative_l2"] = float(np.quantile(vals, .95))
        results.append(rec)
    table = pd.DataFrame(results).sort_values("seed")
    audit = OUT / "audits"
    audit.mkdir(parents=True, exist_ok=True)
    table.to_csv(audit / f"{phase}_per_target.csv", index=False)
    report = {"phase": phase, "runs": len(runs),
              "target_seeds": [42, 43, 44, 45, 46],
              "metric_unit": "independent target run",
              "statistics": {}}
    for col in table.select_dtypes(include=[np.number]):
        if col == "seed":
            continue
        vals = table[col].dropna().to_numpy(dtype=float)
        if len(vals) == 5:
            rng = np.random.default_rng(20261022)
            boot = rng.choice(vals, size=(5000, 5), replace=True).mean(axis=1)
            report["statistics"][col] = {
                "mean": float(vals.mean()), "std": float(vals.std(ddof=1)),
                "bootstrap_95ci": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]}
    (audit / f"{phase}_statistics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare-uncentered")
    for command in ("generate", "verify"):
        p = sub.add_parser(command)
        p.add_argument("--phase", choices=("uncentered", "c09"), required=True)
    args = parser.parse_args()
    if args.command == "prepare-uncentered":
        prepare_uncentered()
    elif args.command == "generate":
        generate(args.phase)
    else:
        verify(args.phase)


if __name__ == "__main__":
    main()
