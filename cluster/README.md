# Clariden Cluster

This directory contains the environment definition and job scripts for running
experiments on the [Clariden](https://docs.cscs.ch/clusters/clariden/) cluster at CSCS.
Clariden is part of the [Alps](https://docs.cscs.ch/alps/) platform and uses
GH200 nodes with the
[Container Engine](https://docs.cscs.ch/software/container-engine/) (CE) as the primary
runtime.

## Files

| File | Purpose |
|---|---|
| `Dockerfile` | Container image built on top of NGC PyTorch 25.06 |
| `constraints-packages.txt` | NGC packages to pin during the image build (prevents torch/CUDA downgrades) |
| `build.sh` | Builds the image interactively (run via `srun --pty bash`) |
| `env.toml` | [Environment Definition File](https://docs.cscs.ch/software/container-engine/run/); defines mounts, env vars, NCCL hooks |


## Environment Design

The environment is split into two layers:

```
Image (.sqsh)  — rebuilt when pyproject.toml or constraints-packages.txt changes
──────────────────────────────────────────────────────────────────────────────────
NGC PyTorch 25.06-py3 (torch, transformers, numpy, ...)
+ project dependencies installed via uv (wandb, hydra, trl, ...)
  with NGC packages pinned via constraints-packages.txt

Mounted at runtime via env.toml
──────────────────────────────────────────────────────────
/iopsstor  ← repo lives here; .env (API keys) lives here too
/capstor   ← HF cache, checkpoints
```

Source code changes never require a rebuild — only `pyproject.toml` or `constraints-packages.txt` changes do.

### How live code editing works

The image installs `amr_fma` along with all dependencies into site-packages. This frozen copy exists only to ensure all dependencies are cached in the image — it is never used at runtime.

`PYTHONPATH` in `env.toml` points to the live mounted repo. Python resolves `PYTHONPATH` entries before site-packages, so the mounted `amr_fma` is what actually gets imported. Running `git pull` at the live path takes effect immediately without any rebuild.


## First-Time Setup

**1. Create the image directory with Lustre striping** ([required by CSCS](https://docs.cscs.ch/software/container-engine/run/)):
```bash
mkdir -p $SCRATCH/ce-images
lfs setstripe -E 4M -c 1 -E 64M -c 4 -E -1 -c -1 -S 4M $SCRATCH/ce-images
```

**2. Create your `.env` file:**
```bash
cp .env.example .env
# then fill in HF_TOKEN and WANDB_API_KEY
```
Keys are loaded automatically at runtime via `python-dotenv` (`amr_fma/core/env.py`). The `.env` file lives in the repo root, which is already mounted into the container via `/iopsstor` — no separate secrets mount needed. The `.env` file is git-ignored.

**3. Build the image:**
```bash
# first exec into an interactive job
srun --account=infra01 --partition=normal --time=01:00:00 --pty bash
# run the build
bash cluster/build.sh
```
Unfortunately, this needs to be done in an interactive job, because sbatch does not enable NAT connections from the node.

> Note: optionally, you can 'borrow' an existing `enroot` image, e.g. from here: `/iopsstor/scratch/cscs/tkwiecinski/ce-images/amr-fma+25.06.sqsh`




## Rebuilding vs. Updating Code

| What changed | Action needed |
|---|---|
| Source code | `git pull` at `/iopsstor/scratch/cscs/$USER/amr-fma` — no rebuild |
| `pyproject.toml` (new dep) | `bash cluster/build.sh` |
| `constraints-packages.txt` | `bash cluster/build.sh` |
| Base image tag | Update `FROM` in `Dockerfile`, then rebuild |

### Notes

To see some insights about working with the cluster, feel free to browse some [tips](../docs/tips_and_tricks.md).

