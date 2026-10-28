#!/usr/bin/env bash
# Lightweight final16 result export (analysis only; no models/checkpoints).
# Run on Rheal: bash scripts/acm_revision/package_final16_analysis.sh A
# Run on Eureka1: bash scripts/acm_revision/package_final16_analysis.sh B
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
case "${1:-}" in
  A|a|uncentered) export PHASE=uncentered SERVER=A ;;
  B|b|c09) export PHASE=c09 SERVER=B ;;
  -h|--help)
    echo "Usage: bash scripts/acm_revision/package_final16_analysis.sh A|B"
    exit 0 ;;
  *) echo "[ERROR] Choose A (uncentered) or B (c09)" >&2; exit 2 ;;
esac
if command -v python3 >/dev/null 2>&1; then PYTHON=python3
elif command -v python >/dev/null 2>&1; then PYTHON=python
else echo "[ERROR] Python 3 required" >&2; exit 2
fi
"$PYTHON" - <<'PY'
import csv
import io
import json
import os
import re
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

root = Path.cwd()
phase = os.environ["PHASE"]
server = os.environ["SERVER"]
base = root / "results/acm_revision/final16_20261022"
run_root = base / phase
try:
    cap_mib = float(os.environ.get("MAX_ANALYSIS_FILE_MIB", "25"))
    assert 0 < cap_mib <= 512
except (ValueError, AssertionError):
    sys.exit("[ERROR] MAX_ANALYSIS_FILE_MIB must be > 0 and <= 512")
cap_bytes = int(cap_mib * 1024 * 1024)
if not run_root.is_dir():
    sys.exit(f"[ERROR] Experiment results missing: {run_root.relative_to(root)}")

# Strict allowlist. Never include tensors, raw updates, model weights,
# checkpoints, entire logs, user datasets or client sample manifests.
allowed_suffixes = {".csv", ".json", ".yaml", ".yml", ".tsv"}
run_names = ("summary.json", "round_metrics.csv", "update_metadata.csv",
             "capture_manifest.json", "resolved_config.json")
selected = {}
skipped = []

def keep(path):
    if not path.is_file() or path.is_symlink() or path.suffix.lower() not in allowed_suffixes:
        return
    try:
        rel = path.relative_to(root).as_posix()
        size = path.stat().st_size
    except (ValueError, OSError):
        return
    if size > cap_bytes:
        skipped.append((rel, f"oversize {size} bytes"))
        return
    selected[rel] = path

for folder in sorted(run_root.iterdir()):
    if folder.is_dir() and not folder.is_symlink():
        for name in run_names:
            keep(folder / name)

audit = base / "audits"
if audit.is_dir():
    keep(audit / f"{phase}_statistics.json")
    keep(audit / f"{phase}_per_target.csv")
    audit_dirs = [audit / f"{phase}_attacks"]
    if phase == "uncentered":
        audit_dirs.append(audit / "matched_update")
    for folder in audit_dirs:
        if folder.is_dir():
            for path in folder.rglob("*"):
                keep(path)

configs = root / "configs/acm_revision/generated/final16_20261022" / phase
if configs.is_dir():
    for path in configs.rglob("*"):
        keep(path)
if phase == "uncentered":
    keep(base / "preparation/uncentered_contrast_bases.json")
for path in (
    root / "configs/acm_revision/hateful_memes.yaml",
    root / "results/acm_revision/defense70_prep/locked_params.yaml",
    root / "results/acm_revision/headfix_prep/locked_headfix.yaml",
):
    keep(path)

expected = [("shadow_train", 142), ("shadow_train", 143),
            ("shadow_val", 242)] + [("target", s) for s in range(42, 47)]
rows = []
for pop, seed in expected:
    matcher = re.compile(rf"__{re.escape(pop)}__.*__seed{seed}__")
    folders = sorted(d for d in run_root.iterdir()
                     if d.is_dir() and matcher.search(d.name))
    if len(folders) != 1:
        rows.append((pop, seed, "MISSING" if not folders else "AMBIGUOUS",
                     "", "", "", ";".join(d.name for d in folders)))
        continue
    folder = folders[0]
    summary = {}
    capture = {}
    try:
        summary_path, manifest_path = folder / "summary.json", folder / "capture_manifest.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if manifest_path.is_file():
            capture = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    highest = 0
    metrics = folder / "round_metrics.csv"
    if metrics.is_file():
        try:
            with metrics.open(encoding="utf-8", newline="") as fp:
                for item in csv.DictReader(fp):
                    if item.get("round"):
                        highest = max(highest, int(float(item["round"])))
        except (OSError, ValueError):
            pass
    rounds = summary.get("rounds", "")
    state = capture.get("status", "")
    complete = state == "PASS" and str(rounds) == "150" and highest == 150
    rows.append((pop, seed, "PASS" if complete else "PARTIAL/NOT_VERIFIED",
                 highest or "", rounds, state, folder.name))

if not selected:
    sys.exit("[ERROR] No analysis files found; archive not created.")

exports = root / "exports"
exports.mkdir(exist_ok=True)
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
dest = exports / f"aslwf_final16_server{server}_{stamp}_analysis.tar.gz"
if dest.exists():
    sys.exit(f"[ERROR] Archive already exists: {dest}")

status_text = io.StringIO()
writer = csv.writer(status_text, lineterminator="\n")
writer.writerow(("population", "seed", "status", "latest_evaluated_round",
                 "summary_rounds", "capture_status", "run_directory"))
writer.writerows(rows)

def add_text(tar, name, value):
    payload = value.encode("utf-8")
    info = tarfile.TarInfo(name)
    info.size, info.mode = len(payload), 0o644
    info.mtime = int(datetime.now(timezone.utc).timestamp())
    tar.addfile(info, io.BytesIO(payload))

try:
    with tarfile.open(dest, "w:gz") as tar:
        included = []
        for rel, path in sorted(selected.items()):
            try:
                if path.stat().st_size > cap_bytes:
                    skipped.append((rel, "changed size above limit during packaging"))
                    continue
                tar.add(path, arcname=rel, recursive=False)
                included.append(rel)
            except OSError as exc:
                skipped.append((rel, f"could not read during packaging: {exc}"))
        info = (
            f"ASLWF final16 | Server {server} | phase={phase}\n"
            f"Created UTC: {datetime.now(timezone.utc).isoformat()}\n"
            f"Small CSV/JSON/YAML/TSV only, per-file limit {cap_mib:g} MiB.\n"
            "Includes per-run metrics, client L2 metadata, final CP/CPT audit,"
            " target uncertainty and plans when available.\n"
            "Excludes raw client data, model weights, updates/*.npz,"
            " checkpoints/*.pt, big logs and client sample manifests.\n"
            "See META/RUN_STATUS.csv for 8 expected runs and PASS verification.\n"
            "Results may be partial if packing while training is in progress.\n"
            "Each server must be exported separately; outputs do not sync automatically.\n"
        )
        add_text(tar, "META/README_ANALYSIS.txt", info)
        add_text(tar, "META/RUN_STATUS.csv", status_text.getvalue())
        add_text(tar, "META/INCLUDED_FILES.txt", "\n".join(included) + "\n")
        add_text(tar, "META/SKIPPED_FILES.txt",
                 "\n".join(f"{name}\t{reason}" for name, reason in skipped) + "\n")
except Exception:
    dest.unlink(missing_ok=True)
    raise

print(f"[PACKAGE DONE] {dest}")
print(f"[INCLUDED] {len(included)} compact analysis files")
print(f"[STATUS] {sum(row[2] == 'PASS' for row in rows)}/8 training runs verified complete")
print(f"[SIZE] {dest.stat().st_size / 1024**2:.2f} MiB")
print("[SAFE] Original training results and checkpoints were not altered")
PY
