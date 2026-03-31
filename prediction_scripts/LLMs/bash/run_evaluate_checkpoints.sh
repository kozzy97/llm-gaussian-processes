#!/bin/bash
#SBATCH --job-name=gp-eval
#SBATCH --output=../../logs/gp-grpo-unsloth/eval/gp_grpo_%A_%a.out
#SBATCH --error=../../logs/gp-grpo-unsloth/eval/gp_grpo_%A_%a.err
#SBATCH --array=1-8:1
# #SBATCH -p gpu_p
# #SBATCH --qos gpu_normal
# #SBATCH --gres=gpu:1
# #SBATCH --constraint=v100_32gb
#SBATCH -p normal
#SBATCH --gres=gpu:full:1
# #SBATCH --mem=40G
#SBATCH -t 24:00:00
#SBATCH --nice=10000
#SBATCH --cpus-per-task=20

base_dir="/gp-icl/unsloth/grpo/"

models=(
    # "unsloth/Qwen3-8B-unsloth-bnb-4bit"
    "unsloth/Qwen3-14B-unsloth-bnb-4bit"
    "unsloth/Qwen3-32B-unsloth-bnb-4bit"
)

length_scales=(1 8)
noise_level_mean=(0.0)
noise_level_sd=(0.01)

checkpoints=(
    "checkpoint-5000"
    "checkpoint-10000"
    "checkpoint-15000"
    "checkpoint-20000"
)

# Index sizes
num_models=${#models[@]}
num_length_scales=${#length_scales[@]}
num_noise_distributions=${#noise_level_mean[@]}
num_checkpoints=${#checkpoints[@]}

index=$((SLURM_ARRAY_TASK_ID - 1))

model_index=$(( index / (num_length_scales * num_noise_distributions * num_checkpoints) ))
ls_index=$(( (index / (num_noise_distributions * num_checkpoints)) % num_length_scales ))
dist_index=$(( (index / num_checkpoints) % num_noise_distributions ))
checkpoint_index=$(( index % num_checkpoints ))

# Fetch parameters
current_model=${models[$model_index]}
current_length_scale=${length_scales[$ls_index]}
current_noise_level_mean=${noise_level_mean[$dist_index]}
current_noise_level_sd=${noise_level_sd[$dist_index]}
current_checkpoint=${checkpoints[$checkpoint_index]}

# Construct path
model_path="${base_dir}${current_model}-ls-${current_length_scale}-mu-${current_noise_level_mean}-sd-${current_noise_level_sd}/${current_checkpoint}"

cd /gp-icl/unsloth

echo "Running evaluation with model: $current_model, length scale: $current_length_scale, noise level mean: $current_noise_level_mean, noise level sd: $current_noise_level_sd, checkpoint: $current_checkpoint"
echo "Model path: $model_path"

conda init bash
source ~/.bashrc

export PATH="~/miniconda3/envs/unsloth_env/bin:$PATH"

conda activate unsloth_env

python evaluate_qwen3.py \
    --model_hf_path $model_path \
    --data_file_path "../data/dataset_1d_mse_se_matern_ls_1_8.json" \
    --save_path "../data/GRPO"
