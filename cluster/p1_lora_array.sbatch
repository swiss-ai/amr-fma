#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --job-name=p1-lora
#SBATCH --output=/iopsstor/scratch/cscs/%u/amr-fma/cluster/log/%x-%A-%a.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/amr-fma/cluster/log/%x-%A-%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1
#SBATCH --array=0-3

set -euo pipefail
trap 'echo "Job failed at $(date)" >&2' ERR

REPO=/iopsstor/scratch/cscs/$USER/amr-fma
LOG_DIR="$REPO/cluster/log"
mkdir -p "$LOG_DIR"

echo "Commit: $(git -C $REPO rev-parse HEAD)"
echo "Branch: $(git -C $REPO rev-parse --abbrev-ref HEAD)"
cd "$REPO"

# Array index maps to: domain x seed
# 0 -> medical seed 42
# 1 -> medical seed 7
# 2 -> code seed 42
# 3 -> code seed 7

DOMAINS=(medical medical code code)
SEEDS=(42 7 42 7)
DATASETS=(chatdoctor chatdoctor python_code python_code)

DOMAIN=${DOMAINS[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
EXPERIMENT_NAME="p1-llama3-${DOMAIN}-s${SEED}"

echo "Task ${SLURM_ARRAY_TASK_ID}: domain=${DOMAIN} seed=${SEED} dataset=${DATASET}"

srun --environment="$REPO/cluster/env.toml" \
    python scripts/run_lora_sft.py \
        model=llama3_8b \
        dataset="${DATASET}" \
        optimization=lora_full \
        runtime=gpu \
        run.domain="${DOMAIN}" \
        run.seed="${SEED}" \
        run.phase=P1 \
        run.run_id="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" \
        run.experiment_name="${EXPERIMENT_NAME}"
