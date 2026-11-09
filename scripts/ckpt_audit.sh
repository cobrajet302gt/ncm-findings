#!/usr/bin/env bash
# Checkpoint-selection audit raised by partner review: in-train chunk peaked ~2-4k steps and
# decayed; stage-2 was built on the FINAL codec. Retrain 4.5k steps reusing cached BPE+segments,
# save every 500, then held-out-eval EVERY checkpoint (CPU) at both window sizes.
set -u
cd /home/cobrajet/ncm-lab
export CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f train_gen_rus[t].py >/dev/null; do sleep 20; done
sleep 3
CORPUS=/home/cobrajet/ncm-lab/data/rust/corpus STEPS=4500 BS=32 AMUL=2 SAVE_EVERY=500 \
  OUT=/home/cobrajet/ncm-lab/artifacts/rustcodec-4k \
  .venv/bin/python train_codec.py 2>&1 | grep -vE "UserWarning|Consider|W1109" | tee artifacts/rustcodec-4k-run.log
echo "=== CKPT AUDIT (held-out, CPU) ==="
for ck in artifacts/rustcodec-4k/tokenizer_*.pt; do
  n=$(basename "$ck" .pt)
  for W in 64 128; do
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=6 BATCH=8 DEV=cpu METRIC_SEG=$W \
      CK=/home/cobrajet/ncm-lab/artifacts/rustcodec-4k CK_NAME="$n.pt" \
      .venv/bin/python rust_codec_eval.py 2>/dev/null | grep HELDOUT | sed "s/^/$n W$W /"
  done
done
# final tokenizer.pt of the 4k run too
for W in 64 128; do
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=6 BATCH=8 DEV=cpu METRIC_SEG=$W \
    CK=/home/cobrajet/ncm-lab/artifacts/rustcodec-4k \
    .venv/bin/python rust_codec_eval.py 2>/dev/null | grep HELDOUT | sed "s/^/tokenizer_final W$W /"
done
echo "AUDIT-DONE"
