#!/usr/bin/env bash
set -u

echo "============================================================"
echo " GPU OOM owner/process check"
echo "============================================================"
echo "Host: $(hostname)"
echo "User: $(whoami)"
echo "Time: $(date '+%F %T')"
echo

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[ERROR] nvidia-smi not found."
  exit 1
fi

echo "[1] GPU memory summary"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total,utilization.gpu --format=csv,noheader || true

echo
echo "[2] Compute processes reported by NVIDIA"
TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null > "$TMP" || true
if [[ ! -s "$TMP" ]]; then
  echo "No GPU compute process reported."
else
  while IFS=',' read -r gpu_uuid pid pname used; do
    pid="$(echo "$pid" | xargs)"
    gpu_uuid="$(echo "$gpu_uuid" | xargs)"
    pname="$(echo "$pname" | xargs)"
    used="$(echo "$used" | xargs)"
    owner="$(ps -o user= -p "$pid" 2>/dev/null | xargs || true)"
    ppid="$(ps -o ppid= -p "$pid" 2>/dev/null | xargs || true)"
    etime="$(ps -o etime= -p "$pid" 2>/dev/null | xargs || true)"
    cmd="$(ps -o args= -p "$pid" 2>/dev/null || true)"
    [[ -z "$owner" ]] && owner="<unknown/ended>"
    echo "PID=$pid owner=$owner used=${used}MiB elapsed=${etime:-?} GPU=$gpu_uuid"
    echo "  PPID=${ppid:-?} process=$pname"
    echo "  CMD=$cmd"
  done < "$TMP"
fi

echo
echo "[3] Current user's ACM run-fl processes"
ps -u "$(whoami)" -o pid,ppid,etime,%cpu,%mem,args 2>/dev/null | grep 'src.acm_revision.cli run-fl' | grep -v grep || echo "None"

echo
echo "[4] Possible orphan ACM run-fl processes (PPID=1)"
ps -u "$(whoami)" -o pid=,ppid=,etime=,args= 2>/dev/null | awk '$2==1 && /src\.acm_revision\.cli run-fl/ {print}' || true

echo
echo "NOTE: Do NOT kill a PID unless owner is your own account and it is confirmed stale."
echo "============================================================"
