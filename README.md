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

# 5. run a LoRA smoke-run (tiny-gpt2, no GPU required)
python scripts/run_lora_sft.py model=tinygpt2 runtime=cpu \
    run.run_id=test run.experiment_name=test

```

You also need to prepare an env file under `.env` in the repo root. You can use the `.env.example` for reference.

For setup on Alps cluster please refer to: [cluster setup](cluster/README.md).

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
│   │   ├── training_config.py         # Typed config sections (DatasetConfig, LoraConfig, …)
│   │   └── lora_sft.py                # LoRA SFT trainer (TRL SFTTrainer + ManifestCallback)
│   │
│   ├── eval/                          # Evaluation of capability + AMR metrics (planned)
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
core  <--  eval
core  <--  interpretability
```

`eval` and `interpretability` never import `fma` directly; they only use `core` abstractions (manifests, RunPaths, checkpoint loaders).
