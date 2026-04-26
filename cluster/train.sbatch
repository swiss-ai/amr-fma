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

mkdir -p $REPO/cluster/log

echo "Commit: $(git -C $REPO rev-parse HEAD)"
echo "Branch: $(git -C $REPO rev-parse --abbrev-ref HEAD)"
cd "$REPO"


srun --environment="$REPO/cluster/env.toml" \
    uv run python scripts/run_lora_sft.py \
        --config-name simple_lora
