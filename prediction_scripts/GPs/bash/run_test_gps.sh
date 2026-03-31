#!/bin/bash
#SBATCH --job-name=likelihood
#SBATCH --output=logs/gp-prediction/%A_%a.out
#SBATCH --error=logs/likelihood-prediction/%A_%a.err
#SBATCH --array=1-8:1
#SBATCH --mail-type=ALL

#SBATCH -p cpu_p
#SBATCH --qos cpu_normal
#SBATCH --mem=40G
#SBATCH -t 12:00:00
#SBATCH --nice=1000
#SBATCH --cpus-per-task=20

# ── define hyperparameter grid ────────────────────────────────────────
kernels=(
    "matern12"
    "matern32"
    "matern52"
    "SE"
)
length_scales=(1 8)

num_kernels=${#kernels[@]}
num_length_scales=${#length_scales[@]}

index=$(( SLURM_ARRAY_TASK_ID - 1 ))

kernel_index=$(( index / num_length_scales ))
ls_index=$(( index % num_length_scales ))

current_kernel=${kernels[$kernel_index]}
current_length_scale=${length_scales[$ls_index]}

# ── go to project directory ─────────────────────────────────────────────────
cd /llm-gaussian-processes/prediction_scripts/GPs

echo "[$(date)] Running task #${SLURM_ARRAY_TASK_ID} → kernel=$current_kernel, ls=$current_length_scale"

# ── load conda and activate env ────────────────────────────────────────────
source ~/miniconda3/etc/profile.d/conda.sh
conda init bash
source ~/.bashrc

export PATH="~/miniconda3/envs/gpy_env/bin:$PATH"
conda activate gpy_env

python test_gps.py \
    --kernel $current_kernel \
    --lengthscale $current_length_scale
