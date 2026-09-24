#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
IMAGE_SIZE="${IMAGE_SIZE:-384}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
NUM_EPOCHS="${NUM_EPOCHS:-200}"
TASK_NAME="${TASK_NAME:-CBI_MedSAM_sessile_Kvasir}"

CUDA_VISIBLE_DEVICES="${GPU}" python main.py \
  --method cbi_medsam \
  --model_type vit_b \
  --checkpoint ./sam_ckp/sam_vit_b_01ec64.pth \
  --data_path ./dataset/sessile-Kvasir \
  --work_dir ./work_dir \
  --task_name "${TASK_NAME}" \
  --image_size "${IMAGE_SIZE}" \
  --label_size "${IMAGE_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --num_epochs "${NUM_EPOCHS}" \
  --use_amp
