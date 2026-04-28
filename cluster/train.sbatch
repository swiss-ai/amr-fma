#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --time=4:00:00
#SBATCH --job-name=amr-fma-test
#SBATCH --output=/iopsstor/scratch/cscs/%u/amr-fma/cluster/log/%x-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/amr-fma/cluster/log/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1

set -euo pipefail
trap 'echo "Job failed at $(date)" >&2' ERR

REPO=/iopsstor/scratch/cscs/$USER/amr-fma
LOG_DIR="$REPO/cluster/log"
JOB_LOG="$LOG_DIR/${SLURM_JOB_NAME:-amr-fma-test}-${SLURM_JOB_ID:-manual}.log"

mkdir -p "$LOG_DIR"

echo "Logging to $JOB_LOG"

echo "Commit: $(git -C $REPO rev-parse HEAD)"
echo "Branch: $(git -C $REPO rev-parse --abbrev-ref HEAD)"
cd "$REPO"

# In-training eval is on by default (chatdoctor sets dataset.eval_samples=512).
# Override per run if needed:
#   dataset.eval_samples=null   # disable
#   dataset.eval_samples=256    # change held-out size
# Eval runs every runtime.logging_steps and produces eval/loss in wandb.
srun --environment="$REPO/cluster/env.toml" \
    python scripts/run_lora_sft.py \
        model=llama3_8b \
        dataset=chatdoctor \
        runtime=gpu \
        run.run_id=${SLURM_JOB_ID:-manual} \
        run.experiment_name=${SLURM_JOB_NAME:-amr-fma-test}
