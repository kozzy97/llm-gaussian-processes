#!/bin/bash
#SBATCH --job-name=gp-eval
#SBATCH --output=logs/llama_eval/%A_%a.out
#SBATCH --error=logs/llama_eval/%A_%a.err
#SBATCH --array=1-1:1
#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100_80gb
#SBATCH --mem=80G
#SBATCH -t 24:00:00
#SBATCH --nice=10000
#SBATCH --cpus-per-task=20

models=(
    "unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit"
    # "unsloth/Llama-3.1-8B-Instruct-unsloth-bnb-4bit"
    # "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit"
)

index=$((SLURM_ARRAY_TASK_ID - 1))

current_model=${models[$index]}

cd /llm-gaussian-processes

echo "Running evaluation with model: $current_model"

source /miniconda3/etc/profile.d/conda.sh

conda activate gp_icl_env

python prediction_scripts/LLMs/evaluate_qwen3.py \
    --model_hf_path "$current_model" \
    --data_file_path "./gp_dataset/dataset_1d_mae_se_matern_ls_1_8.json" \
    --save_path "./model_prediction_data/BASE"
