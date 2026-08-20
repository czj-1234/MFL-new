from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from . import core72_plan as base

# Every physical server owns complete setting/concentration cells, so a cell can
# be post-processed locally without copying partial shadow/target runs first.
# Cells are deliberately ordered with modality_exclusive first. Because each
# cell contains eight jobs and the server queues take alternating jobs, both
# GPUs on both servers begin with modality-exclusive training.
SERVER_CELLS = {
    "A": (
        ("modality_exclusive", 0.5),
        ("modality_exclusive", 0.9),
        ("image_only", 0.5),
        ("image_only", 0.9),
        ("text_only", 0.7),
    ),
    "B": (
        ("modality_exclusive", 0.7),
        ("image_only", 0.7),
        ("text_only", 0.5),
        ("text_only", 0.9),
    ),
}


def all_jobs() -> list[base.CoreJob]:
    return base.all_jobs()


def server_jobs(server: str) -> list[base.CoreJob]:
    server = server.upper()
    if server not in SERVER_CELLS:
        raise ValueError(f"Unknown server {server!r}; expected A or B.")

    jobs_by_cell: dict[tuple[str, float], list[base.CoreJob]] = {
        cell: [] for cells in SERVER_CELLS.values() for cell in cells
    }
    for job in all_jobs():
        cell = (job.setting, job.concentration)
        if cell not in jobs_by_cell:
            raise RuntimeError(f"Unassigned Core72 cell: {cell}")
        jobs_by_cell[cell].append(job)

    ordered: list[base.CoreJob] = []
    for cell in SERVER_CELLS[server]:
        ordered.extend(jobs_by_cell[cell])
    return ordered


def queue_jobs(server: str, queue: int) -> list[base.CoreJob]:
    if queue not in (0, 1):
        raise ValueError("queue must be 0 or 1")
    return server_jobs(server)[queue::2]


def _modality_first(queue: list[base.CoreJob]) -> bool:
    seen_non_modality = False
    for job in queue:
        if job.setting != "modality_exclusive":
            seen_non_modality = True
        elif seen_non_modality:
            return False
    return bool(queue) and queue[0].setting == "modality_exclusive"


def validate_plan() -> dict:
    base_report = base.validate_plan()
    jobs = all_jobs()
    failures: list[str] = []

    all_cells = {
        (setting, concentration)
        for setting in base.SETTINGS
        for concentration in base.CONCENTRATIONS
    }
    assigned_cells = [cell for cells in SERVER_CELLS.values() for cell in cells]
    if set(assigned_cells) != all_cells:
        failures.append(
            f"cell assignment mismatch: missing={sorted(all_cells - set(assigned_cells))}, "
            f"extra={sorted(set(assigned_cells) - all_cells)}"
        )
    if len(assigned_cells) != len(set(assigned_cells)):
        failures.append("a setting/concentration cell is assigned to both servers")

    assigned_jobs = server_jobs("A") + server_jobs("B")
    expected_ids = {job.job_id for job in jobs}
    assigned_ids = [job.job_id for job in assigned_jobs]
    if set(assigned_ids) != expected_ids:
        failures.append("server job assignment does not cover the exact Core72 matrix")
    if len(assigned_ids) != len(set(assigned_ids)):
        failures.append("a Core72 job is assigned more than once")

    queues = {
        f"server_{server}_gpu_queue_{queue}": queue_jobs(server, queue)
        for server in ("A", "B")
        for queue in (0, 1)
    }
    for name, queue in queues.items():
        if not _modality_first(queue):
            failures.append(f"{name} is not modality-exclusive-first")

    counts = {
        "total_jobs": len(jobs),
        "server_A_jobs": len(server_jobs("A")),
        "server_B_jobs": len(server_jobs("B")),
        **{name: len(queue) for name, queue in queues.items()},
        "modality_exclusive_jobs": sum(
            job.setting == "modality_exclusive" for job in jobs
        ),
        "by_setting": dict(Counter(job.setting for job in jobs)),
        "by_concentration": {
            str(key): value
            for key, value in sorted(Counter(job.concentration for job in jobs).items())
        },
    }

    report = {
        "status": "PASS" if not failures else "FAIL",
        "base_core72_status": base_report["status"],
        "execution_order": "modality_exclusive_first_on_all_four_gpu_queues",
        "server_cells": {
            server: [f"{setting}@{concentration}" for setting, concentration in cells]
            for server, cells in SERVER_CELLS.items()
        },
        "queue_first_jobs": {
            name: queue[0].job_id if queue else None for name, queue in queues.items()
        },
        "counts": counts,
        "failures": failures,
    }
    if failures:
        raise RuntimeError("; ".join(failures))
    return report


def validate_data(config_path: str | Path) -> dict:
    # The data matrix is unchanged; reuse the exhaustive real-partitioner check
    # from Core72 for all 72 jobs.
    return base.validate_data(config_path)


def _print_tsv(jobs: Iterable[base.CoreJob]) -> None:
    for job in jobs:
        print(
            "\t".join(
                [
                    job.setting,
                    str(job.concentration),
                    job.population,
                    str(job.seed),
                    str(job.rounds),
                    str(job.num_clients),
                    str(job.samples_per_client),
                    job.job_id,
                ]
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and schedule the modality-exclusive-first Core72 matrix"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate")
    validate.add_argument("--config", default="configs/acm_revision/hateful_memes.yaml")
    validate.add_argument("--check-data", action="store_true")
    validate.add_argument("--output", default=None)

    jobs_parser = sub.add_parser("jobs")
    jobs_parser.add_argument("--server", choices=["A", "B", "a", "b"], default=None)
    jobs_parser.add_argument("--queue", type=int, choices=[0, 1], default=None)
    jobs_parser.add_argument("--json", action="store_true")

    cells_parser = sub.add_parser("cells")
    cells_parser.add_argument("--server", choices=["A", "B", "a", "b"], required=True)

    args = parser.parse_args()
    if args.command == "validate":
        report = {"plan": validate_plan()}
        if args.check_data:
            report["data"] = validate_data(args.config)
        text = json.dumps(report, ensure_ascii=False, indent=2)
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(text, encoding="utf-8")
        print(text)
        return

    if args.command == "cells":
        for setting, concentration in SERVER_CELLS[args.server.upper()]:
            print(f"{setting}\t{concentration}")
        return

    jobs = all_jobs()
    if args.server:
        jobs = server_jobs(args.server)
    if args.queue is not None:
        if not args.server:
            raise SystemExit("--queue requires --server")
        jobs = queue_jobs(args.server, args.queue)
    if args.json:
        print(json.dumps([asdict(job) | {"job_id": job.job_id} for job in jobs], indent=2))
    else:
        _print_tsv(jobs)


if __name__ == "__main__":
    main()
