#!/usr/bin/env bash
# Refuse final16 federated runs if either selected CUDA device cannot execute torch.
# This must run BEFORE creating new training output or shadow contrast updates.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

GPU_FIRST="${GPU0:-0}"
GPU_SECOND="${GPU1:-1}"
if [[ "$GPU_FIRST" == "$GPU_SECOND" ]]; then
  echo "[FATAL CUDA] GPU0 and GPU1 must refer to different physical devices." >&2
  exit 2
fi
command -v nvidia-smi >/dev/null || {
  echo "[FATAL CUDA] nvidia-smi is missing; cannot start formal GPU training." >&2
  exit 2
}
command -v python >/dev/null || {
  echo "[FATAL CUDA] Python is missing from the activated conda environment." >&2
  exit 2
}
for gpu_id in "$GPU_FIRST" "$GPU_SECOND"; do
  nvidia-smi -i "$gpu_id" --query-gpu=name --format=csv,noheader || {
    echo "[FATAL CUDA] Requested physical GPU $gpu_id does not exist." >&2
    exit 2
  }
  echo "[CUDA PREFLIGHT] Testing PyTorch GPU=$gpu_id"
  if ! CUDA_VISIBLE_DEVICES="$gpu_id" python - <<'PY'
import sys
try:
    import torch
    print(f"[CUDA PREFLIGHT] torch={torch.__version__} built_cuda={torch.version.cuda}", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False (driver/runtime may be incompatible)")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected exactly one selected GPU, got {torch.cuda.device_count()}")
    x = torch.ones(1, device="cuda:0")
    torch.cuda.synchronize()
    print(f"[CUDA PREFLIGHT PASS] {torch.cuda.get_device_name(0)} tensor={x.item():.0f}", flush=True)
except Exception as error:
    print(f"[CUDA PREFLIGHT FAILED] {type(error).__name__}: {error}", file=sys.stderr, flush=True)
    raise SystemExit(1)
PY
  then
    echo "[FATAL CUDA] Physical GPU $gpu_id is not usable by this PyTorch environment." >&2
    echo "[FATAL CUDA] Check nvidia-smi driver version and torch.version.cuda." >&2
    echo "[FATAL CUDA] Fix the NVIDIA driver or use a compatible PyTorch build." >&2
    echo "[FATAL CUDA] Do not resume formal runs using CPU-created checkpoints." >&2
    exit 3
  fi
done
echo "[CUDA PREFLIGHT PASS] Both physical GPUs can execute PyTorch CUDA tensors."
