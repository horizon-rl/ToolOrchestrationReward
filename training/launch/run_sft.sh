#!/usr/bin/env bash
# run SFT on 8xH100
# This script can be run from any directory - it will auto-detect paths
# Values are copied from the resolved W&B Hydra dump in sft_config.json

set -x

ulimit -n 65535

# Calculate absolute paths relative to this script's location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
n_gpus_per_node=8

# Model / optimization
MODEL_PATH="/shared/dev/jyc/checkpoints/sft-3-Qwen3-8B"
LEARNING_RATE=0.000001
ENGINE=fsdp

# LoRA / model settings
LORA_RANK=0
LORA_ALPHA=16
TARGET_MODULES="all-linear"

# Data settings
TRAIN_DATA_PATH="$REPO_ROOT/data/bfcl_sft_10_per_workflow.parquet"
VAL_DATA_PATH="$REPO_ROOT/data/eval_sft_100.parquet"
TRAIN_BATCH_SIZE=16
MICRO_BATCH_SIZE_PER_GPU=1
MAX_PROMPT_LENGTH=12384
MAX_RESPONSE_LENGTH=16384

# Trainer settings
PROJECT_NAME='sft_bfcl-tool-agent'
EXPERIMENT_NAME='sft-5-Qwen3-8B-sft-lr0.000001-bs16'
DEFAULT_LOCAL_DIR="$REPO_ROOT/checkpoints/$PROJECT_NAME/$EXPERIMENT_NAME"
CHECKPOINT_SAVE_FREQ=10
TEST_FREQ=20
TOTAL_TRAINING_STEPS=100
TOTAL_EPOCHS=100

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$n_gpus_per_node \
    -m verl.trainer.sft_trainer \
    data.train_files="$TRAIN_DATA_PATH" \
    data.val_files="$VAL_DATA_PATH" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.return_raw_chat=True \
    data.truncation=left \
    data.micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU \
    engine=$ENGINE \
    model.path="$MODEL_PATH" \
    model.lora_rank=$LORA_RANK \
    model.lora_alpha=$LORA_ALPHA \
    model.target_modules=$TARGET_MODULES \
    model.use_remove_padding=True \
    model.enable_gradient_checkpointing=True \
    optim.lr=$LEARNING_RATE \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir="$DEFAULT_LOCAL_DIR" \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=1 \
    trainer.save_freq=$CHECKPOINT_SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_training_steps=$TOTAL_TRAINING_STEPS \
    trainer.total_epochs=$TOTAL_EPOCHS \
    "$@"
