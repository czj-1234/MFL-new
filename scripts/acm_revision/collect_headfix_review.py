from __future__ import annotations

import argparse
import json
import shutil
import socket
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path("results/acm_revision/headfix_r150")
DEFAULT_GROUPS = ("classifier_head", "full_update")
TEXT_FILES = (
    "summary.json",
    "round_metrics.csv",
    "capture_manifest.json",
    "resolved_config.json",
    "client_partition_manifest.json",
    "update_metadata.csv",
)


def _resolve_update_path(path_value: str, repo_root: Path) -> Path:
    p = Path(str(path_value))
    if p.is_absolute() and p.exists():
        return p
    if p.exists():
        return p
    candidate = repo_root / p
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"update file not found: {path_value}")


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _discover_runs(root: Path) -> list[Path]:
    runs = []
    for p in sorted(root.rglob("update_metadata.csv")):
        if (p.parent / "summary.json").exists():
            runs.append(p.parent)
    return runs


def _run_record(run_dir: Path) -> dict:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    record = {
        "run_dir": str(run_dir),
        "run_id": summary.get("run_id"),
        "population": summary.get("population"),
        "seed": summary.get("seed"),
        "setting_name": summary.get("setting_name"),
        "concentration": summary.get("concentration"),
        "rounds": summary.get("rounds"),
    }
    best = summary.get("best") or {}
    for k, v in best.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            record[f"best_{k}"] = v

    metrics_path = run_dir / "round_metrics.csv"
    if metrics_path.exists():
        try:
            metrics = pd.read_csv(metrics_path)
            if not metrics.empty and "round" in metrics.columns:
                final = metrics.sort_values("round").iloc[-1].to_dict()
                for k, v in final.items():
                    if np.isscalar(v):
                        record[f"final_{k}"] = v
        except Exception as exc:
            record["round_metrics_error"] = repr(exc)
    return record


def _feature_row_meta(row: pd.Series, run_dir: Path) -> dict:
    keys = [
        "run_id",
        "population",
        "seed",
        "round",
        "client_id",
        "client_key",
        "dominant_label",
        "modality",
        "setting_name",
        "concentration",
    ]
    out = {k: row.get(k, None) for k in keys}
    out["run_dir"] = str(run_dir)
    out["update_path"] = str(row.get("update_path", ""))
    return out


def collect(root: Path, out_dir: Path, groups: tuple[str, ...]) -> dict:
    repo_root = Path.cwd()
    root = Path(root)
    out_dir = Path(out_dir)

    if not root.exists():
        raise FileNotFoundError(f"headfix result root not found: {root}")

    runs = _discover_runs(root)
    if not runs:
        raise FileNotFoundError(f"no completed/partial headfix run directories found under: {root}")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_records: list[dict] = []
    group_vectors: dict[str, list[np.ndarray]] = {g: [] for g in groups}
    group_meta: dict[str, list[dict]] = {g: [] for g in groups}
    missing_update_files: list[str] = []
    load_errors: list[str] = []

    for run_dir in runs:
        run_records.append(_run_record(run_dir))

        rel = run_dir.relative_to(root)
        dst_run = out_dir / "runs" / rel
        dst_run.mkdir(parents=True, exist_ok=True)
        for name in TEXT_FILES:
            _copy_if_exists(run_dir / name, dst_run / name)

        metadata = pd.read_csv(run_dir / "update_metadata.csv")
        if "update_path" not in metadata.columns:
            load_errors.append(f"{run_dir}: update_metadata.csv has no update_path column")
            continue

        print(
            f"[COLLECT] population={metadata.get('population', pd.Series(['?'])).iloc[0]} "
            f"seed={metadata.get('seed', pd.Series(['?'])).iloc[0]} "
            f"rows={len(metadata)}",
            flush=True,
        )

        for _, row in metadata.iterrows():
            try:
                update_path = _resolve_update_path(str(row["update_path"]), repo_root)
            except FileNotFoundError:
                missing_update_files.append(str(row["update_path"]))
                continue

            try:
                with np.load(update_path, allow_pickle=False) as data:
                    for group in groups:
                        key = f"observed__{group}"
                        if key not in data.files:
                            continue
                        vec = np.asarray(data[key], dtype=np.float32).reshape(-1)
                        group_vectors[group].append(vec)
                        group_meta[group].append(_feature_row_meta(row, run_dir))
            except Exception as exc:
                load_errors.append(f"{update_path}: {repr(exc)}")

    run_index = pd.DataFrame(run_records)
    run_index.to_csv(out_dir / "run_index.csv", index=False)

    feature_report = {}
    for group in groups:
        vectors = group_vectors[group]
        meta = group_meta[group]
        if not vectors:
            feature_report[group] = {"n": 0, "status": "MISSING"}
            continue
        dims = sorted({int(v.size) for v in vectors})
        if len(dims) != 1:
            raise RuntimeError(f"{group}: inconsistent feature dimensions {dims}")
        x = np.stack(vectors).astype(np.float32)
        np.savez_compressed(out_dir / f"observed__{group}.npz", X=x)
        pd.DataFrame(meta).to_csv(out_dir / f"observed__{group}__metadata.csv", index=False)
        feature_report[group] = {
            "n": int(x.shape[0]),
            "dim": int(x.shape[1]),
            "dtype": str(x.dtype),
            "status": "PASS",
        }

    # Include the frozen head-fix manifest and generated formal configs/queues.
    _copy_if_exists(
        Path("results/acm_revision/headfix_prep/locked_headfix.yaml"),
        out_dir / "headfix_prep" / "locked_headfix.yaml",
    )
    _copy_if_exists(
        Path("results/acm_revision/headfix_prep/headfix_candidates.json"),
        out_dir / "headfix_prep" / "headfix_candidates.json",
    )
    cfg_root = Path("configs/acm_revision/generated/headfix")
    if cfg_root.exists():
        shutil.copytree(cfg_root, out_dir / "generated_configs", dirs_exist_ok=True)

    report = {
        "status": "PASS",
        "source_root": str(root),
        "hostname": socket.gethostname(),
        "n_run_dirs_found": len(runs),
        "groups": feature_report,
        "missing_update_file_count": len(missing_update_files),
        "load_error_count": len(load_errors),
        "note": (
            "Review package contains lightweight run metadata plus observed classifier_head "
            "and full_update feature matrices. Large encoder/fusion arrays and checkpoints "
            "are intentionally excluded."
        ),
    }
    (out_dir / "review_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "_MISSING_UPDATE_FILES.txt").write_text(
        "\n".join(missing_update_files),
        encoding="utf-8",
    )
    (out_dir / "_LOAD_ERRORS.txt").write_text(
        "\n".join(load_errors),
        encoding="utf-8",
    )
    return report


def make_archive(out_dir: Path) -> Path:
    archive = out_dir.with_suffix(".tar.gz")
    if archive.exists():
        archive.unlink()
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(out_dir, arcname=out_dir.name)
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Collect a compact HeadFix review package from the runs present on this server. "
            "The package is suitable for upload/review and excludes large checkpoints."
        )
    )
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory. Default: headfix_review_<hostname>",
    )
    parser.add_argument(
        "--groups",
        default=",".join(DEFAULT_GROUPS),
        help="Comma-separated observed surfaces to extract (default: classifier_head,full_update)",
    )
    args = parser.parse_args()

    groups = tuple(x.strip() for x in args.groups.split(",") if x.strip())
    hostname = socket.gethostname().replace("/", "_")
    out_dir = Path(args.out or f"headfix_review_{hostname}")

    report = collect(Path(args.root), out_dir, groups)
    archive = make_archive(out_dir)

    print()
    print("=" * 64)
    print("[HEADFIX REVIEW PACKAGE READY]")
    print(f"Runs found : {report['n_run_dirs_found']}")
    for group, info in report["groups"].items():
        print(f"{group:16s}: {info}")
    print(f"Missing update files: {report['missing_update_file_count']}")
    print(f"Load errors         : {report['load_error_count']}")
    print(f"Archive             : {archive}")
    print(f"Archive size        : {archive.stat().st_size / (1024 * 1024):.1f} MB")
    print("=" * 64)


if __name__ == "__main__":
    main()
