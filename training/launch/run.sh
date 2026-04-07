# run on 8xH100
# This script can be run from any directory - it will auto-detect paths

set -x

ulimit -n 65535

# NOTE: nest_asyncio causes conflicts with Ray's asyncio event loop
# Commenting out to prevent "ValueError: loop argument must agree with lock"
# python3 -c "import nest_asyncio; nest_asyncio.apply()"

# Calculate absolute paths relative to this script's location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_PATH="$SCRIPT_DIR/config"

export VLLM_USE_V1=1

REWARD_FUNCTION_PATH="$REPO_ROOT/verl/verl/utils/reward_score/complex_tool.py"
REWARD_FUNCTION_NAME="compute_score"

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
n_gpus_per_node=8
MODEL_PATH="Qwen/Qwen3-8B"
# MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
# MODEL_PATH="Salesforce/Llama-xLAM-2-8b-fc-r"

# Key hyperparameters for run name
LEARNING_RATE=0.000001

BATCH_SIZE=16
MINI_BATCH_SIZE=16
ALGORITHM=grpo
ROLLOUT_N=16
KL_COEF=0.001 # default 0.001
CHECKPOINT_SAVE_FREQ=10 # -1 to disable, >0 to save every N steps
MAX_CHECKPOINTS_TO_KEEP=2 # null to keep all, N to keep latest N checkpoints

# Reward weights (sum should be <= 1.0)
# Set to 0.0 to disable a component (improves performance by skipping computation)
REWARD_ATOMIC=0.5           # R_atomic_validity: Individual function call correctness
REWARD_ORCH=0.5             # R_outcome_orchestration: Multi-step orchestration with dependencies
REWARD_STATE=0.0            # R_outcome_state: Final state validation [PLACEHOLDER - not implemented]

# Multi-turn tool calling parameters
MAX_USER_TURNS=20  # Max tool execution turns (user messages)
MAX_ASSISTANT_TURNS=20  # Max model generation turns (assistant messages)
MAX_TOOL_RESPONSE_LENGTH=8192  # Max TOKENS in tool response
# Token distribution (Qwen tokenizer): p50=586, p95=11.5K, p99=21K, max=49K
# 8192:  covers 90.4% (allows ~5 tool turns within context window)
# 16384: covers 97.8% (allows ~3 tool turns, may hit context limit)
# Adjust based on model's context window and expected turn count

# Generate reward tag for experiment tracking
# Format: a<weight>o<weight>[s<weight>]
# Examples: "a0.5o0.5", "a1.0", "a0.3o0.7", "a0.3o0.5s0.2"
REWARD_TAG="a${REWARD_ATOMIC}o${REWARD_ORCH}"
# Only append state weight if non-zero
if (( $(echo "$REWARD_STATE > 0" | bc -l) )); then
    REWARD_TAG="${REWARD_TAG}s${REWARD_STATE}"
fi

PROJECT_NAME='rl_bfcl-tool-agent'
EXPERIMENT_NAME=$MODEL_PATH-$ALGORITHM-$REWARD_TAG-n$ROLLOUT_N-lr$LEARNING_RATE-bs$BATCH_SIZE-mbs$MINI_BATCH_SIZE-kl$KL_COEF-Nov27-10-partial

# Data paths - Using new balanced 100-sample training data from ComplexFuncBench eval
# Contains perfect domain balance (20 samples each) with original distractor tools
# TRAIN_DATA_PATH="$REPO_ROOT/data/bfcl_rl_training_100_hack.parquet"
# TEST_DATA_PATH="$REPO_ROOT/data/bfcl_rl_training_100_hack.parquet"

# TRAIN_DATA_PATH="$REPO_ROOT/data/bfcl_rl_training_3_per_workflow.parquet"
# TEST_DATA_PATH="$REPO_ROOT/data/bfcl_rl_training_3_per_workflow.parquet"

TRAIN_DATA_PATH="$REPO_ROOT/data/bfcl_rl_training_10_per_workflow.parquet"
TEST_DATA_PATH="$REPO_ROOT/data/data/eval_rl_100.parquet"

# Alternative: Use 1000-sample dataset (40% Cross-domain, 15% each other domain)
# TRAIN_DATA_PATH="$REPO_ROOT/data/bfcl_rl_training_1000_hack.parquet"
# TEST_DATA_PATH="$REPO_ROOT/data/bfcl_rl_training_1000_hack.parquet"

# # ============================================================================
# # MLflow S3 Sync Configuration (automatic)
# # ============================================================================
# # Automatically syncs mlflow traces to S3 during training
# # Uses project and experiment names from above configuration
# #
# # To disable: export MLFLOW_S3_BUCKET="" before running
# # To override: export MLFLOW_S3_BUCKET="s3://your-bucket/path"
# #
# # S3 organization: s3://bucket/project/experiment-timestamp/mlruns.db

# # Set default S3 bucket (override by exporting MLFLOW_S3_BUCKET before running)
# : ${MLFLOW_S3_BUCKET:="s3://your-bucket/mlflow_traces"}
# : ${MLFLOW_SYNC_FREQUENCY_STEPS:=10}

# if [[ -n "${MLFLOW_S3_BUCKET}" ]]; then
#     echo "========================================="
#     echo "MLflow S3 Sync Enabled"
#     echo "========================================="
#     echo "S3 Bucket: $MLFLOW_S3_BUCKET"
#     echo "Project: $PROJECT_NAME"
#     echo "Experiment: $EXPERIMENT_NAME"
#     echo ""

#     # Export configuration for sync daemon
#     export MLFLOW_S3_BUCKET
#     export MLFLOW_SYNC_FREQUENCY_STEPS
#     export MLFLOW_PROJECT_NAME="$PROJECT_NAME"
#     export MLFLOW_EXPERIMENT_NAME="$EXPERIMENT_NAME"
#     export MLFLOW_SYNC_LOG_FILE="$REPO_ROOT/verl/mlflow_sync.log"

#     # Start sync daemon in background
#     SYNC_SCRIPT="$REPO_ROOT/verl/sync_mlflow_to_s3.sh"
#     if [[ -f "$SYNC_SCRIPT" ]]; then
#         nohup bash "$SYNC_SCRIPT" > /dev/null 2>&1 &
#         SYNC_DAEMON_PID=$!
#         echo "✓ MLflow sync daemon started (PID: $SYNC_DAEMON_PID)"
#         echo "  Log file: $MLFLOW_SYNC_LOG_FILE"
#         echo "  Sync frequency: Every $MLFLOW_SYNC_FREQUENCY_STEPS steps"
#         echo "  S3 destination: $MLFLOW_S3_BUCKET/$MLFLOW_PROJECT_NAME/$MLFLOW_EXPERIMENT_NAME-[timestamp]/"
#         echo ""

#         # Ensure daemon stops when this script exits
#         cleanup_sync_daemon() {
#             if [[ -n "${SYNC_DAEMON_PID:-}" ]] && kill -0 "$SYNC_DAEMON_PID" 2>/dev/null; then
#                 echo ""
#                 echo "Stopping MLflow sync daemon..."
#                 kill -TERM "$SYNC_DAEMON_PID" 2>/dev/null || true
#                 wait "$SYNC_DAEMON_PID" 2>/dev/null || true
#                 echo "✓ MLflow sync daemon stopped"
#             fi
#         }
#         trap cleanup_sync_daemon EXIT
#     else
#         echo "⚠ Warning: sync_mlflow_to_s3.sh not found at $SYNC_SCRIPT"
#         echo "  S3 sync will not be enabled"
#         echo ""
#     fi
# else
#     echo "ℹ MLflow S3 sync disabled (MLFLOW_S3_BUCKET not set)"
#     echo ""
# fi

# ============================================================================

# python preprocess_bfcl_for_rl.py \
#     --input data/bfcl_training_8.pkl \
#     --output data/bfcl_rl_training_8.parquet \
#     --tool-schema-path environment/booking_api.json \
#     --filter-success

# data.filter_overlong_prompts=True \

# # change to vllm
# # Disable Ray dashboard to avoid asyncio event loop conflicts
# export RAY_DASHBOARD_HOST=0.0.0.0
# export RAY_DASHBOARD_PORT=0
# export RAY_DISABLE_DASHBOARD=1

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='gsm8k_multiturn_grpo' \
    algorithm.adv_estimator=$ALGORITHM \
    data.train_batch_size=$BATCH_SIZE \
    data.return_raw_chat=True \
    data.truncation='left' \
    data.max_prompt_length=12384 \
    data.max_response_length=16384 \
    data.train_files=$TRAIN_DATA_PATH \
    data.val_files=$TEST_DATA_PATH \
    custom_reward_function.path=$REWARD_FUNCTION_PATH \
    custom_reward_function.name=$REWARD_FUNCTION_NAME \
    custom_reward_function.reward_kwargs.reward_weights.R_atomic_validity=$REWARD_ATOMIC \
    custom_reward_function.reward_kwargs.reward_weights.R_outcome_orchestration=$REWARD_ORCH \
    custom_reward_function.reward_kwargs.reward_weights.R_outcome_state=$REWARD_STATE \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.trace.backend=weave \
    actor_rollout_ref.rollout.trace.token2text=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=$MAX_USER_TURNS \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$MAX_ASSISTANT_TURNS \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=$MAX_TOOL_RESPONSE_LENGTH \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    +trainer.tags="[reward:${REWARD_TAG},algorithm:${ALGORITHM}]" \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=1 \
    trainer.save_freq=$CHECKPOINT_SAVE_FREQ \
    trainer.max_actor_ckpt_to_keep=$MAX_CHECKPOINTS_TO_KEEP \
    trainer.test_freq=20 \
    trainer.total_training_steps=100 \
    trainer.total_epochs=100 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$CONFIG_PATH/bfcl_tool_config.yaml" $@