#!/bin/bash
#SBATCH --job-name=gp-sft_matern
#SBATCH --output=/llm-gaussian-processes/logs/gp-sft-matern/gp_sft_matern_%A_%a.out
#SBATCH --error=/llm-gaussian-processes/logs/gp-sft-matern/gp_sft_matern_%A_%a.err
#SBATCH --array=1-6:1
#SBATCH --nodes=1
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100_80gb
#SBATCH --mem=80G
#SBATCH --cpus-per-task=20
#SBATCH --time=24:00:00
#SBATCH --nice=10000

models=(
    "unsloth/Qwen3-8B-unsloth-bnb-4bit"
    "unsloth/Qwen3-14B-unsloth-bnb-4bit"
    "unsloth/Qwen3-32B-unsloth-bnb-4bit"
)

length_scales=(1 8)
noise_level_mean=(0.0)
noise_level_sd=(0.01)

num_models=${#models[@]}
num_length_scales=${#length_scales[@]}
num_noise_level_distributions=${#noise_level_mean[@]}

index=$((SLURM_ARRAY_TASK_ID - 1))
model_index=$(( index / (num_length_scales * num_noise_level_distributions) ))
ls_index=$(( (index / num_noise_level_distributions) % num_length_scales ))
dist_index=$(( index % num_noise_level_distributions ))

current_model=${models[$model_index]}
current_length_scale=${length_scales[$ls_index]}
current_noise_level_mean=${noise_level_mean[$dist_index]}
current_noise_level_sd=${noise_level_sd[$dist_index]}

save_dir="/llm-gaussian-processes/finetuned_model_saves/matern/qlora_mae/$current_model-ls-$current_length_scale-mu-$current_noise_level_mean-sd-$current_noise_level_sd"

cd /llm-gaussian-processes/prediction_scripts/LLMs/

echo "Running task with model: $current_model and length scale: $current_length_scale and noise level mean: $current_noise_level_mean and noise level sd: $current_noise_level_sd"

echo "Saving to: $save_dir"

source "$HOME/.bashrc"
conda activate unsloth_new

python train_qlora_matern.py \
    --seed 1997 \
    --length_scales $current_length_scale \
    --num_noise_values 2 \
    --noise_level_mean $current_noise_level_mean \
    --noise_level_sd $current_noise_level_sd \
    --model_name $current_model \
    --output_dir $save_dir
