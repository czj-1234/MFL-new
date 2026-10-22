"""Shadow-only matched-original-update L2 distortion audit.

Replays identical raw client updates from original reference checkpoints through
frozen defense configurations. This is NOT a claim about historical 150-round
realized distortions or a privacy score, and never touches target data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import defense70_plan, headfix_plan
from .data_protocol import create_matched_composition_pair, load_json, make_client_specs
from .defense_dispatch import apply_defense_to_state
from .fl import build_runtime, local_train, set_global_seed
from .orchestrator import load_yaml
from .supplement16 import (BASIS, HEAD_LOCK, OLD_LOCK, OUT, OTHER_GROUPS,
                           REF_CONFIG, load_locks, require_file)

CHECKPOINT_ROUNDS = (10, 75, 150)


def exact_distortion(before, raw, defended):
    """Fallback for clipping/noise baselines without a relative-L2 budget."""
    from .defenses import name_in_group
    sums = {key: [0., 0.] for key in ("classifier_head", "full_update")}
    for name in before:
        if not torch.is_floating_point(before[name]):
            continue
        base = before[name].detach().cpu().float()
        delta = raw[name].detach().cpu().float() - base
        perturb = defended[name].detach().cpu().float() - raw[name].detach().cpu().float()
        raw_sq = float(torch.sum(delta.double().square()).item())
        diff_sq = float(torch.sum(perturb.double().square()).item())
        sums["full_update"][0] += raw_sq
        sums["full_update"][1] += diff_sq
        if name_in_group(name, "classifier_head"):
            sums["classifier_head"][0] += raw_sq
            sums["classifier_head"][1] += diff_sq
    return {g: {"raw_l2": np.sqrt(a), "difference_l2": np.sqrt(b),
                "relative_l2": np.sqrt(b)/(np.sqrt(a)+1e-12)}
            for g, (a,b) in sums.items()}


def run(rounds=CHECKPOINT_ROUNDS, client_count=12, with_uncentered=True):
    matched, head = load_locks()
    cfg = load_yaml(require_file(REF_CONFIG))
    if cfg["experiment"]["population"] != "shadow_train":
        raise ValueError("Reference config must use shadow_train")
    base = load_yaml("configs/acm_revision/hateful_memes.yaml")
    reference = Path(matched["provenance"]["reference_run_dir"])
    all_rounds = set(matched["provenance"]["reference_rounds"])
    if not set(rounds).issubset(all_rounds):
        raise ValueError("Audit rounds must come from locked reference checkpoints")
    if not 1 <= client_count <= 12:
        raise ValueError("client_count must be 1..12")
    pool = load_json(require_file(Path(cfg["data"]["pools_dir"]) / "shadow_val.json"))
    client_specs = make_client_specs(12, "modality_exclusive", int(cfg["data"]["num_classes"]))[:client_count]
    # All methods use the EXACT SAME raw update. No method-specific local training.
    ops = {op.op_id: op.defense for op in defense70_plan.operating_points(matched)}
    ops["complete_aslwf"] = headfix_plan._defense_cfg(matched, head)
    if with_uncentered:
        require_file(BASIS)
        variant = headfix_plan._defense_cfg(matched, head)
        for layer in variant["layers"]:
            if layer["group"] in OTHER_GROUPS:
                layer["basis_path"] = str(BASIS)
        ops["uncentered_complete_aslwf"] = variant
    set_global_seed(142)
    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, tokenizer, processor = build_runtime(cfg)
    observations = []
    for r in rounds:
        state = torch.load(require_file(reference / "checkpoints" / f"round_{r:04d}.pt"),
                           map_location="cpu")
        model.load_state_dict(state, strict=True)
        model.to("cpu")
        before = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
        print(f"[AUDIT] round {r}: {client_count} matched raw updates; {len(ops)} defenses", flush=True)
        for spec in client_specs:
            seed = 242 * 1_000_000 + r * 10_000 + spec.client_id
            concentrated, _, _ = create_matched_composition_pair(
                pool=pool, spec=spec, num_classes=int(cfg["data"]["num_classes"]),
                samples_per_client=100, concentrated=.7, balanced=.5, seed=seed)
            raw, _ = local_train(model, concentrated, tokenizer, processor,
                                 spec.modality, cfg, device, seed)
            for method, defense in ops.items():
                defended, metadata = apply_defense_to_state(before, raw, defense, seed=seed+13)
                budget = metadata.get("relative_l2_budget")
                if budget and budget.get("norm_space") == "exact_original_model_update":
                    actual = {g: {
                        "raw_l2": budget["raw_update_l2"][g],
                        "difference_l2": budget["actual_difference_l2"][g],
                        "relative_l2": budget["relative_after"][g]}
                        for g in ("classifier_head", "full_update")}
                else:
                    actual = exact_distortion(before, raw, defended)
                for group in ("classifier_head", "full_update"):
                    observations.append({
                        "method": method, "round": int(r), "client": int(spec.client_id),
                        "group": group, **actual[group],
                        "exact_original_update_space": True,
                        "method_specific_clipping": method.startswith("clipgauss")})
                del defended
            del raw
        del before, state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if not observations:
        raise RuntimeError("Distortion audit produced no results")
    out = OUT / "audits" / "matched_update"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(observations)
    df.to_csv(out / "per_matched_update.csv", index=False)
    groups = []
    for (method, group), frame in df.groupby(["method", "group"]):
        arr = frame["relative_l2"].to_numpy(dtype=float)
        groups.append({
            "method": method, "group": group, "n_update_samples": len(arr),
            "mean_relative_l2": float(np.mean(arr)),
            "median_relative_l2": float(np.median(arr)),
            "p95_relative_l2": float(np.quantile(arr, .95)),
            "mean_raw_l2": float(frame["raw_l2"].mean()),
            "mean_difference_l2": float(frame["difference_l2"].mean()),
        })
    pd.DataFrame(groups).sort_values(["method","group"]).to_csv(out / "summary.csv", index=False)
    metadata = {
        "scope": "frozen shadow reference checkpoints and shadow_val auxiliary clients",
        "not_historical_trajectory_statistics": True,
        "no_target_data": True,
        "rounds": list(rounds),
        "clients_per_round": client_count,
        "same_raw_updates_shared_by_all_defenses": True,
        "contains_method_specific_clip_gaussian": True,
        "original_parameter_space_l2": True,
        "notes": ["This matched-checkpoint audit cannot replace historical per-client L2 measurements.",
                  "Clip-Gaussian uses a different clipping/noise mechanism and is not budget matched.",
                  "No privacy or utility comparisons are inferred from distortion alone."],
    }
    (out / "protocol.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[PASS] actual distortion summary -> {out / 'summary.csv'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", default="10,75,150")
    parser.add_argument("--clients", type=int, default=12)
    parser.add_argument("--without-uncentered", action="store_true")
    args = parser.parse_args()
    rounds = tuple(int(s) for s in args.rounds.split(","))
    run(rounds=rounds, client_count=args.clients, with_uncentered=not args.without_uncentered)


if __name__ == "__main__":
    main()
