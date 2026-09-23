#!/usr/bin/env bash
set -euo pipefail

GPU_IDS="${GPU_IDS:-1,2,3,4}"
COMFYUI_DIR="${COMFYUI_DIR:-$HOME/ComfyUI}"
PORT="${PORT:-8188}"

cat <<EOF
# MiniMax H3 4x V100 environment (review before starting)
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$GPU_IDS
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=1
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export COMFY_MINIMAX_SP_POOL=${COMFY_MINIMAX_SP_POOL:-1}
export COMFY_MINIMAX_SP_FP16_ATTN=${COMFY_MINIMAX_SP_FP16_ATTN:-1}
export COMFY_MINIMAX_SP_FP16_COMM=${COMFY_MINIMAX_SP_FP16_COMM:-1}

# The process-visible IDs become 0..3. Do not pass the physical K620.
cd $COMFYUI_DIR
exec ./venv/bin/python main.py --listen 127.0.0.1 --port $PORT
EOF

