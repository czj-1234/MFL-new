#!/usr/bin/env bash
set -euo pipefail

SERVER="${1:?Usage: bash scripts/acm_revision/watch_defense70_progress.sh A|B}"
SERVER="${SERVER^^}"
REFRESH="${DEFENSE70_PROGRESS_REFRESH:-10}"
JOBS_FILE="${DEFENSE70_JOBS_FILE:-configs/acm_revision/generated/defense70/jobs.tsv}"
LOG_ROOT="${DEFENSE70_LOG_ROOT:-logs/acm_revision/defense70_r150}"
STATE_ROOT="${DEFENSE70_STATE_ROOT:-logs/acm_revision/defense70_r150/state}"
RUNTIME_ROOT="${LOG_ROOT}/runtime"
PREP_STATUS="${DEFENSE70_PREP_STATUS_DIR:-logs/acm_revision/defense70_r150/prep}/status.json"
PREP_LOG="${DEFENSE70_PREP_STATUS_DIR:-logs/acm_revision/defense70_r150/prep}/prepare.log"
TOTAL_ROUNDS=150

if [[ "${SERVER}" != "A" && "${SERVER}" != "B" ]]; then
  echo "SERVER must be A or B" >&2
  exit 2
fi

# During the first launch jobs.tsv does not exist yet by design. Keep the UI
# alive and show the shadow-only preparation stage instead of reporting a false
# scheduler failure. Once the locked 70-job matrix appears, switch to the normal
# round-level dashboard automatically.
while [[ ! -f "${JOBS_FILE}" ]]; do
  printf '\033[2J\033[H'
  echo "============================================================================================"
  echo "DEFENSE70 SERVER ${SERVER} — PREPARATION"
  echo "============================================================================================"
  PIDFILE="${RUNTIME_ROOT}/server${SERVER}.pid"
  if [[ -f "${PIDFILE}" ]]; then
    PID="$(cat "${PIDFILE}" 2>/dev/null || true)"
    if [[ "${PID}" =~ ^[0-9]+$ ]] && kill -0 "${PID}" 2>/dev/null; then
      echo "Scheduler: RUNNING  PID=${PID}"
    else
      echo "Scheduler: NOT RUNNING"
    fi
  else
    echo "Scheduler: STARTING / PID file not written yet"
  fi

  if [[ -f "${PREP_STATUS}" ]]; then
    python - "${PREP_STATUS}" <<'PY'
import json, pathlib, sys
try:
    obj = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
    print(f"Stage:   {obj.get('stage','UNKNOWN')}")
    print(f"Message: {obj.get('message','')}")
except Exception as exc:
    print(f"Stage: status file is being updated ({exc})")
PY
  elif [[ "${SERVER}" == "B" ]]; then
    echo "Stage:   WAITING"
    echo "Message: waiting for Server A to finish the one-time shadow-only preparation"
  else
    echo "Stage:   STARTING"
    echo "Message: waiting for the preparation process to publish its first status"
  fi

  echo "--------------------------------------------------------------------------------------------"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits 2>/dev/null | \
      awk -F',' '{gsub(/ /,"",$0); split($0,a,","); printf "GPU %s: mem %s/%s MiB | util %s%%\n",a[1],a[2],a[3],a[4]}' || true
  fi
  echo "--------------------------------------------------------------------------------------------"
  echo "Formal jobs have not started yet. This is expected on the first launch."
  echo "Preparation log: ${PREP_LOG}"
  echo "Ctrl+C closes ONLY this monitor; the background preparation/training process keeps running."
  echo "Refresh every ${REFRESH}s."
  sleep "${REFRESH}"
done

python - "${SERVER}" "${JOBS_FILE}" "${LOG_ROOT}" "${STATE_ROOT}" "${RUNTIME_ROOT}" "${REFRESH}" "${TOTAL_ROUNDS}" <<'PY'
import csv
import os
import pathlib
import subprocess
import sys
import time
from datetime import datetime

import yaml

server, jobs_file, log_root, state_root, runtime_root, refresh, total_rounds = sys.argv[1:]
refresh = max(2, int(refresh))
total_rounds = int(total_rounds)
jobs_file = pathlib.Path(jobs_file)
log_root = pathlib.Path(log_root)
state_root = pathlib.Path(state_root)
runtime_root = pathlib.Path(runtime_root)


def run_id_from_cfg(cfg):
    exp = cfg.get("experiment", {})
    fed = cfg.get("federated", {})
    model = cfg.get("model", {}).get("architecture", "clip_dual")
    defense = cfg.get("defense", {}).get("name", "none")
    concentration = exp.get("concentration", exp.get("association", "iid"))
    population = exp.get("population", "target")
    return (
        f"{cfg.get('data', {}).get('name','dataset')}__{model}__{population}__{exp['setting_name']}__c{concentration}"
        f"__n{fed['num_clients']}__p{fed.get('participation_rate',1.0)}__{fed.get('aggregation','fedavg')}"
        f"__seed{cfg['seed']}__def-{defense}"
    ).replace("/", "-")


def read_jobs():
    rows = []
    global_idx = 0
    local_idx = 0
    with jobs_file.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.rstrip("\n")
            if not raw or raw.startswith("#"):
                continue
            parts = raw.split("\t")
            if parts[0] == "job_id":
                continue
            if len(parts) != 2:
                continue
            job_id, cfg_path = parts
            belongs = (server == "A" and global_idx % 2 == 0) or (server == "B" and global_idx % 2 == 1)
            if belongs:
                with open(cfg_path, "r", encoding="utf-8") as cf:
                    cfg = yaml.safe_load(cf)
                exp = cfg.get("experiment", {})
                output_root = pathlib.Path(exp.get("output_root", "results/acm_revision/defense70_r150"))
                run_id = run_id_from_cfg(cfg)
                slot = local_idx % 4
                gpu = 0 if slot in (0, 1) else 1
                gpu_slot = 0 if slot in (0, 2) else 1
                rows.append({
                    "job_id": job_id,
                    "config": cfg_path,
                    "run_dir": output_root / run_id,
                    "gpu": gpu,
                    "slot": gpu_slot,
                    "local_index": local_idx,
                    "population": exp.get("population", "target"),
                    "seed": cfg.get("seed"),
                    "defense": cfg.get("defense", {}).get("name", "none"),
                    "op": exp.get("defense70_operating_point", "unknown"),
                })
                local_idx += 1
            global_idx += 1
    return rows


def max_round(metadata_path):
    if not metadata_path.exists() or metadata_path.stat().st_size == 0:
        return 0
    m = 0
    try:
        with metadata_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    m = max(m, int(float(row.get("round", 0))))
                except Exception:
                    pass
    except Exception:
        return 0
    return m


def marker_exists(kind, job_id):
    suffix = "done" if kind == "done" else "failed"
    return (state_root / kind / f"{job_id}.{suffix}").exists()


def server_alive():
    pidfile = runtime_root / f"server{server}.pid"
    if not pidfile.exists():
        return False, None
    try:
        pid = int(pidfile.read_text().strip())
        os.kill(pid, 0)
        return True, pid
    except Exception:
        return False, None


def gpu_summary():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        lines = []
        for line in out.strip().splitlines():
            idx, used, total, util = [x.strip() for x in line.split(",")]
            lines.append(f"GPU {idx}: mem {used}/{total} MiB | util {util}%")
        return lines
    except Exception:
        return ["GPU status unavailable"]


jobs = read_jobs()
if len(jobs) != 35:
    print(f"[ERROR] Server {server} expected 35 jobs, found {len(jobs)}")
    raise SystemExit(2)

try:
    while True:
        data = []
        done = failed = active = starting = pending = 0
        round_sum = 0
        for j in jobs:
            r = max_round(j["run_dir"] / "update_metadata.csv")
            is_done = marker_exists("done", j["job_id"]) or r >= total_rounds
            is_failed = marker_exists("failed", j["job_id"]) and not is_done
            logs = sorted(log_root.glob(f"{j['job_id']}.attempt*.log"))
            has_log = bool(logs)
            if is_done:
                status = "DONE"
                done += 1
                r = total_rounds
            elif is_failed:
                status = "FAILED"
                failed += 1
            elif r > 0:
                status = "RUNNING"
                active += 1
            elif has_log:
                status = "STARTING"
                starting += 1
            else:
                status = "PENDING"
                pending += 1
            round_sum += min(r, total_rounds)
            data.append((status, r, j))

        alive, pid = server_alive()
        pct = 100.0 * round_sum / (len(jobs) * total_rounds)
        print("\033[2J\033[H", end="")
        print("=" * 100)
        print(f"DEFENSE70 SERVER {server} — LIVE PROGRESS     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 100)
        print(f"Scheduler: {'RUNNING' if alive else 'NOT RUNNING'}" + (f"  PID={pid}" if pid else ""))
        print(
            f"Jobs: DONE {done}/35 | RUNNING {active} | STARTING {starting} | "
            f"PENDING {pending} | FAILED {failed}"
        )
        print(f"Round progress: {round_sum}/{len(jobs)*total_rounds} = {pct:5.1f}%")
        for line in gpu_summary():
            print(line)
        print("-" * 100)
        print("ACTIVE / STARTING")
        shown = 0
        for status_name, r, j in data:
            if status_name not in {"RUNNING", "STARTING"}:
                continue
            shown += 1
            bar_n = int(20 * r / total_rounds)
            bar = "#" * bar_n + "-" * (20 - bar_n)
            print(
                f"GPU{j['gpu']}/S{j['slot']}  {status_name:8s}  [{bar}] {r:3d}/{total_rounds}  "
                f"{j['op']} | {j['population']} seed={j['seed']}"
            )
        if shown == 0:
            print("(none yet)")

        print("-" * 100)
        finished = [(s, r, j) for s, r, j in data if s == "DONE"][-5:]
        print("MOST RECENT COMPLETED / CURRENT COMPLETED SET (up to 5 shown)")
        if finished:
            for _, r, j in finished:
                print(f"DONE  {r:3d}/{total_rounds}  {j['op']} | {j['population']} seed={j['seed']}")
        else:
            print("(none yet)")

        print("-" * 100)
        print(f"Refresh every {refresh}s. Ctrl+C closes ONLY this monitor; background training keeps running.")
        print(f"Re-open anytime: bash scripts/acm_revision/watch_defense70_progress.sh {server}")
        sys.stdout.flush()
        time.sleep(refresh)
except KeyboardInterrupt:
    print("\nProgress monitor closed. Defense70 background preparation/training was NOT stopped.")
PY
