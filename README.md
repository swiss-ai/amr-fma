# AMR-FMA: Temporal Safety Drift under Frontier Model Adaptation

> This is work in progress, really early stage of development

This repository contains code to study **temporal drift of anthropomorphic misalignment risk (AMR)** during **Frontier Model Adaptation (FMA)** of large language models.

The initial focus is:

- 8B-parameter models from several families (e.g. Apertus, OLMo, Qwen, Llama, DeepSeek).
- FMA methods: LoRA+SFT, full SFT, and SDPO.
- Domains: medical advice and code generation.
- Metrics:
  - General capability: MMLU-Pro, BigBench-Hard, IFEval, Alpaca Eval 2.0, Arena-Hard.
  - AMR: harmfulness, emergent misalignment, evaluation awareness, deception, sycophancy, shutdown resistance.

Later phases will explore **active interpretability** (probe-based penalties, steering, activation clamping) and **scaling to 32B models** (e.g. OLMo-2-32B).

---

## Getting started

This is a basic setup on a local machine:

```bash

# 1. Clone and enter
git clone https://github.com/swiss-ai/amr-fma
cd amr-fma

# 2. Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Create env and install
uv sync

# 4. set your own env variables
cp .env.example .env

# 5a. run a LoRA smoke-run (tiny-gpt2, no GPU required)
python scripts/run_lora_sft.py model=tinygpt2 runtime=cpu \
    run.run_id=test run.experiment_name=test

# 5b. or run a full SFT run
python scripts/run_full_sft.py model=tinygpt2 runtime=cpu \
    run.run_id=test run.experiment_name=test

```

You also need to prepare an env file under `.env` in the repo root. You can use the `.env.example` for reference.

For setup on Alps cluster please refer to: [cluster setup](cluster/README.md).

---

## Training observability

Logged to wandb and persisted to `trainer_state.json` (per checkpoint and at the run-dir root):

- TRL defaults: `train/{loss,learning_rate,grad_norm,epoch}`.
- `eval/loss` at each checkpoint step when an eval split is configured.
- `eval/perplexity`, `eval/token_accuracy`, `eval/loss_std` from `compute_metrics`.
- The run manifest as a versioned wandb Artifact at run end.

### In-training evaluation

A held-out slice (`dataset.eval_samples`, default `512` for `chatdoctor`) is passed to `SFTTrainer`. Eval runs at the same steps as checkpoints; the split seed is fixed independently of `run.seed` so results are comparable across seeds.

```bash
python scripts/run_lora_sft.py dataset.eval_samples=256 ...   # change size
python scripts/run_lora_sft.py dataset.eval_samples=null ...  # disable
```

---

## Configuration

Configs live in `configs/` and are composed at runtime via [Hydra](https://hydra.cc/). Each subdirectory is a config group; `config.yaml` sets the defaults. Override any field on the command line: `optimization.learning_rate=1e-4`. To select from LoRA or full SFT training, simply run different script: `scripts/run_lora_sft.py` or `scripts/run_full_sft.py` respectively.

**`model/`** — model family and LoRA target modules. One file per model (e.g. `llama3_8b.yaml`, `tinygpt2.yaml`).

**`lora/`** — LoRA rank (`r`), scaling (`alpha`), and dropout. Defaults: `r=16`, `alpha=32`. Ignored by full SFT method

**`dataset/`** — dataset name, split, which text field to use, and how many samples to load (`max_samples`, `eval_samples`). `eval_samples` carves out a held-out slice used for in-training evaluation.

**`optimization/`** — all training-loop hyperparameters: epochs, batch size, gradient accumulation, learning rate, scheduler, warmup, weight decay, and grad norm clipping.

**`sequence/`** — tokenization settings: `max_length` and whether to use packing (concatenating short examples to fill context windows, reducing wasted compute).

**`runtime/`** — machine-level flags: `gpu.yaml` enables bf16 and gradient checkpointing; `cpu.yaml` disables them for local smoke-runs.

**`checkpointing/`** — controls how many checkpoints are saved during a run (`num_checkpoints`) and how many to keep on disk (`save_total_limit`). Checkpoints are spaced evenly across training steps. Each time a checkpoint is written, the adapter weights are dumped to disk and `manifest.yaml` is updated atomically. The manifest is a YAML file that lives at the run root and records run metadata (model, dataset, seed, git commit, hyperparams) plus an entry for every checkpoint — path, step, timestamp. It's the single source of truth for downstream evaluation and interpretability phases.

**`evaluation/`** — controls in-training evaluation. `steps.yaml` enables it and sets `eval_steps` (how often to evaluate, in steps). `disabled.yaml` turns it off entirely. Evaluation runs the held-out `eval_samples` split through the model and logs `loss`, `perplexity`, and `token_accuracy` to wandb.

---

## High-level design

We separate the codebase into three conceptual layers:

- `core`: shared abstractions for runs, manifests, checkpoints, models, and configuration.
- `fma`: training and adaptation (P1, and resuming for P2/P3). The SFT module is built on top of the TRL library.
- `eval`: general and AMR-specific evaluation pipelines (planned).
- `interpretability`: caching activations, training probes, and running mitigation interventions (P2).

Only the **FMA layer** mutates model weights and writes checkpoints. Evaluation and interpretability are read-only on top of the recorded checkpoints.

Configuration and experiment orchestration are handled via **Hydra**, and experiment tracking/logging via **Weights & Biases (wandb)**.

---

## Repository structure

Current layout (subject to iteration):

```text
amr-fma/
├── amr_fma/                           # Installable Python package
│   ├── core/                          # Shared contracts + utilities
│   │   ├── paths.py                   # RunPaths: canonical P1/P2/P3 directory layout
│   │   ├── manifest.py                # RunManifest dataclass + YAML serialisation
│   │   ├── checkpointing.py           # Atomic manifest writes, checkpoint scheduling
│   │   ├── models.py                  # Loading base models/adapters, device/dtype placement
│   │   └── env.py                     # BASE_OUTPUT_DIR and other env var helpers
│   │
│   ├── fma/                           # Phase 1: adaptation (LoRA+SFT, full SFT, SDPO)
│   │   ├── callbacks.py               # Trainer callbacks (ManifestCallback, ...)
│   │   ├── training_config.py         # Typed config sections (DatasetConfig, LoraConfig, …)
│   │   └── lora_sft.py                # LoRA SFT trainer (TRL SFTTrainer)
│   │
│   │
│   └── interpretability/              # Phase 2: active interpretability / mitigation (planned)
│
├── configs/                           # Hydra configuration tree (runtime, not Python)
│   ├── config.yaml                    # Top-level defaults list + experiment identity fields
│   ├── model/                         # Model family + LoRA target modules (one file per model)
│   ├── dataset/                       # Dataset source and sampling settings
│   ├── optimization/                  # Training-loop hyperparameters
│   ├── sequence/                      # Tokenisation and packing settings
│   ├── checkpointing/                 # Checkpoint frequency and retention
│   └── runtime/                       # Machine settings: gpu.yaml / cpu.yaml
│
├── scripts/                           # Thin entry points (Hydra + env setup)
│   └── run_lora_sft.py                # LoRA SFT training entry point
│
├── cluster/                           # SLURM job scripts for Alps/CSCS (GH200 nodes)
│
├── tests/
│   └── test_lora_sft_smoke.py         # End-to-end smoke test (no real model weights needed)
│
├── README.md
├── TODO.md                            # Roadmap/milestones
├── pyproject.toml                     # uv/pip dependencies
├── .env.example                       # BASE_OUTPUT_DIR, WANDB_API_KEY, HF_HOME
└── .github/workflows/ci.yml           # Ruff + pytest enforcement on PRs
```

Dependency direction is **one-way**:

```text
core  <--  fma
core  <--  interpretability
```

`interpretability` never import `fma` directly; it only uses `core` abstractions (manifests, RunPaths, checkpoint loaders).
