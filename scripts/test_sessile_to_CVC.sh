#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
IMAGE_SIZE="${IMAGE_SIZE:-384}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MODEL_CKPT="${MODEL_CKPT:-}"

if [[ -z "${MODEL_CKPT}" || ! -f "${MODEL_CKPT}" ]]; then
  echo "Set MODEL_CKPT to the checkpoint trained on sessile-Kvasir." >&2
  exit 2
fi

CUDA_VISIBLE_DEVICES="${GPU}" python main.py \
  --method cbi_medsam \
  --model_type vit_b \
  --checkpoint ./sam_ckp/sam_vit_b_01ec64.pth \
  --data_path ./dataset/CVC \
  --image_size "${IMAGE_SIZE}" \
  --label_size "${IMAGE_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --test_only \
  --resume "${MODEL_CKPT}" \
  --use_amp
