# ToolOrchestrationReward: Multi-Step Tool Orchestration with Constrained Data Synthesis and Graduated Rewards

[![arXiv](https://img.shields.io/badge/arXiv-2603.24709-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2603.24709)

Official code for **"Training LLMs for Multi-Step Tool Orchestration with Constrained Data Synthesis and Graduated Rewards"**.

This repository provides:

- **Constrained data synthesis** for BFCL / BookingAPI-style multi-turn tool use (workflow templates, cache-aligned generation).
- **Graduated rewards** for RL: \(R_{\text{atomic}}\) (per-call validity) and \(R_{\text{orch}}\) (orchestration across steps), wired for **[VERL](https://github.com/volcengine/verl)** GRPO-style training.
- **Evaluation** on [ComplexFuncBench](https://huggingface.co/datasets/zai-org/ComplexFuncBench), including an optional BFCL-consistent evaluator aligned with training.

---

## 1. Dependencies

### Core (this repo)

- **Python** ≥ 3.13 (see `pyproject.toml`).
- Install this repo in editable mode:

```bash
pip install -e .
```

### RL stack (training)

Training is built on **[verl v0.5.0](https://github.com/verl-project/verl/tree/v0.5.0)**. Clone or install that version of `verl` first, along with its dependencies (Ray, vLLM, etc.). Confirm you can run a basic `verl` example before launching ToolOrchestrationReward training.

```
git clone --branch v0.5.0 https://github.com/verl-project/verl.git verl
cd verl
pip3 install -e ".[vllm]"
```

### BFCL dependency

BFCL comes from the Gorilla repository: **[ShishirPatil/gorilla](https://github.com/ShishirPatil/gorilla/tree/main)**.

This repo keeps the BookingAPI tool schema locally at:

- `environment/booking_api.json`

For BFCL execution / validation, clone Gorilla and move its `berkeley-function-call-leaderboard` directory to this repo as `BFCL` so the existing `BFCL/bfcl_eval/...` imports continue to work:

```bash
git clone https://github.com/ShishirPatil/gorilla.git
mv gorilla/berkeley-function-call-leaderboard BFCL
```

After that, the repository should contain paths such as:

- `BFCL/bfcl_eval/eval_checker/multi_turn_eval/...`

The Hydra tool config reads the schema from `${oc.env:REPO_ROOT}/environment/booking_api.json` (`training/launch/config/bfcl_tool_config.yaml`), while BFCL runtime imports still resolve from `BFCL/bfcl_eval/...`.

### Integrating ToolOrchestrationReward Code into VERL

`training/launch/run.sh` loads the reward from **inside your VERL tree**:

- `REPO_ROOT/verl/verl/utils/reward_score/complex_tool.py`

After cloning `verl` next to this project (recommended layout: `<ToolOrchestrationReward>/verl/` as the `verl` repo root), copy the graduated reward implementation from this repo into VERL:

```bash
# From the ToolOrchestrationReward repo root (adjust if your VERL path differs)
cp training/reward/complex_tool.py verl/verl/utils/reward_score/complex_tool.py
```

Copy **`training/tools/BFCLClientManager.py`** and **`training/tools/bfcl_tool.py`** into the matching locations under your VERL install (same module paths as upstream VERL BFCL tools: `verl.tools...`). Upstream VERL already provides `base_tool`, `schemas`, and MCP client utilities; this repo’s files extend or replace the BFCL client for the BookingAPI cache setup.

### Environment variables

- Set **`REPO_ROOT`** to this repository’s root (the launch script exports it from `training/launch/run.sh`; override if you run Hydra/VERL manually).
- **`WANDB_API_KEY`** — training logs to Weights & Biases (`trainer.logger` includes `"wandb"` in `run.sh`).

### API keys

```bash
cp .env.example .env
```

Edit `.env` with at least one LLM provider key (see comments in `.env.example`). Optional: **RapidAPI** key only if you run live cache expansion (`environment/expand_cache.py`).

---

## 2. Repository layout

```
data_synthesis/          # Training data generation & preprocessing
environment/             # Cache-backed BookingAPI environment; optional cache expansion
training/                # Reward module, tool wrappers (copy into VERL), launch scripts
evaluation/              # ComplexFuncBench evaluation
utils/                   # Shared utilities (LLM helpers, prompts, logging)
data/                    # Workflow templates (`workflow_templates.json`, 107 patterns)
```

---

## 3. Data preparation

### External data

- Download **[ComplexFuncBench](https://huggingface.co/datasets/zai-org/ComplexFuncBench)** and place the JSONL as:

  `data/ComplexFuncBench.jsonl`

  (Used by generation scripts and evaluation.)

### Generate BFCL-aligned training pickle

From the **repository root**:

```bash
python data_synthesis/generate_bfcl_training_data.py \
  --queries-per-workflow 10 \
  --model gpt-5.1 \
  --num-distractors 2 \
  --output data/bfcl_training_10_per_workflow.pkl \
  --verbose
```

- Requires **`BFCL/`** on `PYTHONPATH` / install so `bfcl_eval` imports resolve.
- Default workflow file: `data/workflow_templates.json`.

### Preprocess to VERL parquet

```bash
python data_synthesis/preprocess_bfcl_for_rl.py \
  --input data/bfcl_training_10_per_workflow.pkl \
  --output data/bfcl_rl_training_10_per_workflow.parquet \
  --tool-schema-path environment/booking_api.json \
  --filter-success

python data_synthesis/preprocess_bfcl_for_rl.py \
  --input data/eval_100.pkl \
  --output data/eval_rl_100.parquet \
  --tool-schema-path environment/booking_api.json \
  --filter-success
```

`--tool-schema-path` defaults to the same path if omitted.

If you need the SFT data, you just need to run

```bash
python data_synthesis/preprocess_bfcl_for_sft.py \
  --input data/bfcl_training_10_per_workflow.pkl \
  --output data/bfcl_sft_training_10_per_workflow.parquet \
  --tool-schema-path environment/booking_api.json \
  --filter-success

python data_synthesis/preprocess_bfcl_for_sft.py \
  --input data/eval_100.pkl \
  --output data/eval_sft_100.parquet \
  --tool-schema-path environment/booking_api.json \
  --filter-success
```

---

## 4. Launching RL training

Example (Linux; assumes **8 GPUs** — adjust `CUDA_VISIBLE_DEVICES` / `n_gpus_per_node` in `training/launch/run.sh` for your hardware):

```bash
cd training/launch && bash run.sh
```

The script invokes `python3 -m verl.trainer.main_ppo` with Hydra config `gsm8k_multiturn_grpo` from your `verl v0.5.0` install and overrides for BFCL multi-turn rollout, custom reward weights, and `bfcl_tool_config.yaml`.

### Main knobs (see `run.sh`)

| Area | Variables / flags |
|------|-------------------|
| Model | `MODEL_PATH` (e.g. `Qwen/Qwen3-8B`) |
| Algorithm | `ALGORITHM` (e.g. `grpo`) |
| Batch | `BATCH_SIZE`, `MINI_BATCH_SIZE`, `ROLLOUT_N` |
| Reward mix | `REWARD_ATOMIC`, `REWARD_ORCH`, `REWARD_STATE` |
| Multi-turn caps | `MAX_USER_TURNS`, `MAX_ASSISTANT_TURNS`, `MAX_TOOL_RESPONSE_LENGTH` |
| Data | `TRAIN_DATA_PATH`, `TEST_DATA_PATH` |
| Logging | `PROJECT_NAME`, `EXPERIMENT_NAME`, W&B |

---

## 5. Evaluation

From the **repository root** (editable install resolves `utils` / `evaluators`):

```bash
python evaluation/evaluation.py \
  --model_name "your-model" \
  --input_file data/ComplexFuncBench.jsonl \
  --use-bfcl
```

- **`--use-bfcl`** — deterministic BFCL execution aligned with training. This is the supported evaluation path for this open-source release. It requires `BFCL/` at repo root and the `verl` package (or a local `<repo>/verl` clone on `PYTHONPATH`).

---

## 6. (Optional) Expand BookingAPI cache

Live API calls require **`RAPID_API_KEY`** (see `.env.example`). They also require the live RapidAPI helper / ComplexFuncBench metadata used by `environment/booking_api.py`; cached mode works without these extras:

```bash
python environment/expand_cache.py \
  --functions Search_Hotels,Search_Flights \
  --samples-per-function 50
```

Training and evaluation can use the **existing** checked-in cache without calling live APIs.

---

## 7. `PYTHONPATH` and roots

When running outside `run.sh`, ensure:

- **`REPO_ROOT`** points at this repo’s root (for `${oc.env:REPO_ROOT}` in YAML).
- **`BFCL/`** exists under that root.
- The **`verl`** package is importable (install or add your VERL clone root to `PYTHONPATH`).
- The **`bfcl_eval`** package is importable (for a local clone, add `BFCL/` itself to `PYTHONPATH`).

Example:

```bash
export REPO_ROOT=/path/to/ToolOrchestrationReward
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/BFCL:${PYTHONPATH}"
```

---

## 📖 Citation

```bibtex
@misc{cheng2026training,
      title={Training LLMs for Multi-Step Tool Orchestration with Constrained Data Synthesis and Graduated Rewards},
      author={
        {Jiayang Cheng} and
        {Xin Liu} and
        {Zhihan Zhang} and
        {Haoyang Wen} and
        {Zixuan Zhang} and
        {Qingyu Yin} and
        {Shiyang Li} and
        {Priyanka Nigam} and
        {Bing Yin} and
        {Chao Zhang} and
        {Yangqiu Song}
      },
      year={2026},
      eprint={2603.24709},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
}
```
