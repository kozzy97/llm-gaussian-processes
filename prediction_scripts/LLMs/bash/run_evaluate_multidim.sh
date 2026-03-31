#!/bin/bash
#SBATCH --job-name=gp-eval
#SBATCH --output=/logs/multidim_eval/%A_%a.out
#SBATCH --error=/logs/multidim_eval/%A_%a.err
#SBATCH --array=1-1:1
#SBATCH -p gpu_p
#SBATCH --qos gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100_80gb
#SBATCH --mem=80G
#SBATCH -t 24:00:00
#SBATCH --nice=10000
#SBATCH --cpus-per-task=20

## Load parameters for jobs

models=(
    # "unsloth/Qwen3-8B-unsloth-bnb-4bit"
    # "unsloth/Qwen3-14B-unsloth-bnb-4bit"
    "unsloth/Qwen3-32B-unsloth-bnb-4bit"
)

datasets=(
    # "./gp_dataset/dataset_2d_mae_se_matern_ls_1_8.json"
    # "./gp_dataset/dataset_3d_mae_se_matern_ls_1_8.json"
    "./gp_dataset/dataset_4d_mae_se_matern_ls_1_8.json"
)

num_models=${#models[@]}
num_datasets=${#datasets[@]}
total_combinations=$((num_models * num_datasets))

index=$((SLURM_ARRAY_TASK_ID - 1))
model_index=$((index / num_datasets))
dataset_index=$((index % num_datasets))

current_model=${models[$model_index]}
current_data=${datasets[$dataset_index]}

cd /llm-gaussian-processes

echo "=========================================="
echo "SLURM Job Array ID: $SLURM_ARRAY_JOB_ID"
echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running combination $((SLURM_ARRAY_TASK_ID)) of $total_combinations"
echo "Model ($model_index): $current_model"
echo "Dataset ($dataset_index): $current_data"
echo "=========================================="

source /miniconda3/etc/profile.d/conda.sh

conda activate gp_icl_env

python prediction_scripts/LLMs/evaluate_multidim.py \
    --model_hf_path "$current_model" \
    --data_file_path "$current_data" \
    --save_path "./model_prediction_data/MULTI-DIM/BASE"

echo "Job completed at $(date)"