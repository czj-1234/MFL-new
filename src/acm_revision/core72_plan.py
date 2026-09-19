from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .data_protocol import load_json, make_client_specs, partition_clients_strict
from .orchestrator import load_yaml


SETTINGS = ("image_only", "text_only", "modality_exclusive")
CONCENTRATIONS = (0.5, 0.7, 0.9)
ROUNDS = 150
NUM_CLIENTS = 12
BATCH_SIZE = 16
LOCAL_EPOCHS = 1
TARGET_SEEDS = (42, 43, 44, 45, 46)
SHADOW_TRAIN_SEEDS = (142, 143)
SHADOW_VAL_SEEDS = (242,)
POPULATION_SEEDS = {
    "shadow_train": SHADOW_TRAIN_SEEDS,
    "shadow_val": SHADOW_VAL_SEEDS,
    "target": TARGET_SEEDS,
}
SAMPLES_PER_CLIENT = {
    "shadow_train": 200,
    "shadow_val": 100,
    "target": 200,
}

# Keep every setting/concentration cell on one physical server, so post-processing
# never needs partial cell results from two servers.
SERVER_CELLS = {
    "A": (
        ("image_only", 0.5),
        ("image_only", 0.7),
        ("image_only", 0.9),
        ("text_only", 0.5),
        ("text_only", 0.7),
    ),
    "B": (
        ("text_only", 0.9),
        ("modality_exclusive", 0.5),
        ("modality_exclusive", 0.7),
        ("modality_exclusive", 0.9),
    ),
}


@dataclass(frozen=True)
class CoreJob:
    setting: str
    concentration: float
    population: str
    seed: int
    rounds: int = ROUNDS
    num_clients: int = NUM_CLIENTS
    samples_per_client: int = 0

    @property
    def job_id(self) -> str:
        ctag = str(self.concentration).replace(".", "p")
        return (
            f"{self.setting}__c{ctag}__{self.population}__seed{self.seed}"
            f"__n{self.num_clients}__s{self.samples_per_client}__r{self.rounds}"
        )


def all_jobs() -> list[CoreJob]:
    jobs: list[CoreJob] = []
    for setting in SETTINGS:
        for concentration in CONCENTRATIONS:
            for population in ("shadow_train", "shadow_val", "target"):
                for seed in POPULATION_SEEDS[population]:
                    jobs.append(
                        CoreJob(
                            setting=setting,
                            concentration=concentration,
                            population=population,
                            seed=int(seed),
                            samples_per_client=SAMPLES_PER_CLIENT[population],
                        )
                    )
    return jobs


def server_jobs(server: str) -> list[CoreJob]:
    server = server.upper()
    if server not in SERVER_CELLS:
        raise ValueError(f"Unknown server {server!r}; expected A or B.")
    owned = set(SERVER_CELLS[server])
    return [j for j in all_jobs() if (j.setting, j.concentration) in owned]


def queue_jobs(server: str, queue: int) -> list[CoreJob]:
    if queue not in (0, 1):
        raise ValueError("queue must be 0 or 1")
    jobs = server_jobs(server)
    return [job for index, job in enumerate(jobs) if index % 2 == queue]


def _expected_counts() -> dict:
    jobs = all_jobs()
    by_population = Counter(job.population for job in jobs)
    by_setting = Counter(job.setting for job in jobs)
    by_concentration = Counter(job.concentration for job in jobs)
    by_cell = Counter((job.setting, job.concentration) for job in jobs)
    return {
        "total_jobs": len(jobs),
        "by_population": dict(sorted(by_population.items())),
        "by_setting": dict(sorted(by_setting.items())),
        "by_concentration": {str(k): v for k, v in sorted(by_concentration.items())},
        "by_cell": {f"{k[0]}@{k[1]}": v for k, v in sorted(by_cell.items())},
        "server_A_jobs": len(server_jobs("A")),
        "server_B_jobs": len(server_jobs("B")),
        "server_A_queue_0": len(queue_jobs("A", 0)),
        "server_A_queue_1": len(queue_jobs("A", 1)),
        "server_B_queue_0": len(queue_jobs("B", 0)),
        "server_B_queue_1": len(queue_jobs("B", 1)),
    }


def validate_plan() -> dict:
    jobs = all_jobs()
    failures: list[str] = []

    ids = [job.job_id for job in jobs]
    if len(jobs) != 72:
        failures.append(f"expected 72 jobs, found {len(jobs)}")
    if len(ids) != len(set(ids)):
        failures.append("duplicate job IDs detected")

    expected_per_cell = (
        len(SHADOW_TRAIN_SEEDS) + len(SHADOW_VAL_SEEDS) + len(TARGET_SEEDS)
    )
    counts = Counter((job.setting, job.concentration) for job in jobs)
    for setting in SETTINGS:
        for concentration in CONCENTRATIONS:
            value = counts[(setting, concentration)]
            if value != expected_per_cell:
                failures.append(
                    f"{setting}@{concentration}: expected {expected_per_cell} jobs, found {value}"
                )

    assigned = server_jobs("A") + server_jobs("B")
    assigned_ids = [job.job_id for job in assigned]
    if set(assigned_ids) != set(ids):
        missing = sorted(set(ids) - set(assigned_ids))
        extra = sorted(set(assigned_ids) - set(ids))
        failures.append(f"server assignment mismatch: missing={missing}, extra={extra}")
    if len(assigned_ids) != len(set(assigned_ids)):
        failures.append("a job is assigned to both servers")

    report = {
        "status": "PASS" if not failures else "FAIL",
        "constants": {
            "settings": list(SETTINGS),
            "concentrations": list(CONCENTRATIONS),
            "rounds": ROUNDS,
            "num_clients": NUM_CLIENTS,
            "batch_size": BATCH_SIZE,
            "local_epochs": LOCAL_EPOCHS,
            "population_seeds": {k: list(v) for k, v in POPULATION_SEEDS.items()},
            "samples_per_client": dict(SAMPLES_PER_CLIENT),
        },
        "counts": _expected_counts(),
        "failures": failures,
    }
    if failures:
        raise RuntimeError("; ".join(failures))
    return report


def validate_data(config_path: str | Path) -> dict:
    cfg = load_yaml(config_path)
    pools_dir = Path(cfg["data"]["pools_dir"])
    pool_files = {
        "shadow_train": pools_dir / "shadow_train.json",
        "shadow_val": pools_dir / "shadow_val.json",
        "target": pools_dir / "target.json",
    }
    missing = [str(path) for path in pool_files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing strict pool files: {missing}")

    pools = {name: load_json(path) for name, path in pool_files.items()}
    num_classes = int(cfg["data"]["num_classes"])
    results = []
    failures = []

    for job in all_jobs():
        specs = make_client_specs(job.num_clients, job.setting, num_classes)
        try:
            client_data, manifest = partition_clients_strict(
                pool=pools[job.population],
                specs=specs,
                num_classes=num_classes,
                concentration=job.concentration,
                samples_per_client=job.samples_per_client,
                seed=job.seed,
                partition_mode="fixed",
            )
            total = sum(len(values) for values in client_data.values())
            expected = job.num_clients * job.samples_per_client
            if total != expected or manifest.get("unique_assigned") != expected:
                raise AssertionError(
                    f"assigned={total}, unique={manifest.get('unique_assigned')}, expected={expected}"
                )
            results.append(
                {
                    "job_id": job.job_id,
                    "status": "PASS",
                    "pool_size": len(pools[job.population]),
                    "assigned": total,
                }
            )
        except Exception as exc:
            failures.append({"job_id": job.job_id, "error": repr(exc)})
            results.append({"job_id": job.job_id, "status": "FAIL", "error": repr(exc)})

    report = {
        "status": "PASS" if not failures else "FAIL",
        "config": str(config_path),
        "pool_sizes": {name: len(values) for name, values in pools.items()},
        "n_jobs_checked": len(results),
        "n_failures": len(failures),
        "failures": failures,
        "jobs": results,
    }
    if failures:
        raise RuntimeError(
            f"Core72 data-capacity validation failed for {len(failures)} jobs; "
            f"first={failures[0]}"
        )
    return report


def _print_tsv(jobs: Iterable[CoreJob]) -> None:
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
    parser = argparse.ArgumentParser(description="Generate and validate the strict 72-run core matrix")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate")
    p.add_argument("--config", default="configs/acm_revision/hateful_memes.yaml")
    p.add_argument("--check-data", action="store_true")
    p.add_argument("--output", default=None)

    p = sub.add_parser("jobs")
    p.add_argument("--server", choices=["A", "B", "a", "b"], default=None)
    p.add_argument("--queue", type=int, choices=[0, 1], default=None)
    p.add_argument("--json", action="store_true")

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
