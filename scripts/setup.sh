#!/usr/bin/env bash
# NCM sandbox setup on GPU1 side-lane (judge stays live; NCM is small)
LOG=/home/cobrajet/ncm-lab/setup.log
exec >> "$LOG" 2>&1
echo "=== ncm-lab setup $(date -Is) ==="
cd /home/cobrajet/ncm-lab || exit 1
df -h /home/cobrajet | tail -1
free -g | head -2

echo "[step] clone"
[ -d ncm ] || git clone --depth 1 https://github.com/HazyResearch/ncm ncm

echo "[step] venv"
if command -v uv >/dev/null 2>&1; then uv venv --python 3.12 --seed --clear .venv; else rm -rf .venv; python3 -m venv .venv; fi

echo "[step] pip install (torch cu128 for sm_120 + repo reqs)"
PIP=/home/cobrajet/ncm-lab/.venv/bin/python
$PIP -m pip install -U pip wheel
$PIP -m pip install -U "huggingface_hub[cli]"
$PIP -m pip install -r ncm/requirements.txt \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple

echo "[step] torch sanity on GPU1"
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PIP - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
free, total = torch.cuda.mem_get_info()
print("GPU:", torch.cuda.get_device_name(0), f"free={free/2**20:.0f}MiB total={total/2**20:.0f}MiB")
x = torch.randn(1024, 1024, device="cuda") @ torch.randn(1024, 1024, device="cuda")
torch.cuda.synchronize()
print("matmul ok", float(x.sum()))
EOF

echo "[step] checkpoints"
HF=/home/cobrajet/ncm-lab/.venv/bin/hf
$HF download hazyresearch/ncm-checkpoints --local-dir /home/cobrajet/ncm-lab/artifacts --include "sudoku-hard/*" || \
  $HF download hazyresearch/ncm-checkpoints --local-dir /home/cobrajet/ncm-lab/artifacts --include "*sudoku*hard*"
$HF download hazyresearch/ncm-checkpoints --local-dir /home/cobrajet/ncm-lab/artifacts --include "xsum/*"
$HF download hazyresearch/ncm-checkpoints --local-dir /home/cobrajet/ncm-lab/artifacts --include "tinystories/*"

echo "[step] artifacts on disk"
find /home/cobrajet/ncm-lab/artifacts -type f -printf "%s %p\n" 2>/dev/null | sort -n

echo "[step] eval configs available"
ls ncm/config/eval/ 2>/dev/null

echo "[done] $(date -Is)"
