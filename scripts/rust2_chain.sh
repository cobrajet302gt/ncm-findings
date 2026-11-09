#!/usr/bin/env bash
# Stage-2 rebuild on the held-out PEAK codec (tokenizer_2500: W64 96.6%, text-exact 88.3%),
# per checkpoint audit. Waits for the in-flight B1 (rust1 x 20k-final codec baseline),
# retrains the generator on 2500-codec codes, then re-runs B1 against the new pair.
set -u
cd /home/cobrajet/ncm-lab
export CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f eval_rust_b1.p[y] >/dev/null; do sleep 15; done
sleep 3
echo "=== rust2 generator on PEAK codec (tokenizer_2500) ==="
TRAIN_STEPS=6000 BS=24 LR=3e-4 OUT=/home/cobrajet/ncm-lab/train/rust2 \
  CK=/home/cobrajet/ncm-lab/artifacts/rustcodec-4k CK_NAME=tokenizer_2500.pt \
  .venv/bin/python train_gen_rust.py 2>&1 | grep -vE "UserWarning|Consider|parameter count" | tee train/rust2-run.log
echo "=== B1 on rust2 x peak codec ==="
N=128 B=32 GEN=/home/cobrajet/ncm-lab/train/rust2 \
  CK=/home/cobrajet/ncm-lab/artifacts/rustcodec-4k CK_NAME=tokenizer_2500.pt \
  .venv/bin/python eval_rust_b1.py 2>&1 | grep -vE "UserWarning|Consider|parameter count" | tee eval/rust-b1-rust2-run.log
echo "RUST2-DONE"
