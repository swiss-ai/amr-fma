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

# 5. run the experiment
uv run amr-fma-dummy

```

You also need to prepare an env file under `.env` in the repo root. You can use the `.env.example` for reference.


For setup on Alps cluster please refer to: [cluster setup](cluster/README.md).

---

## High-level design

We separate the codebase into three conceptual layers:

- `core`: shared abstractions for runs, manifests, checkpoints, models, and configuration.
- `fma`: training and adaptation (P1, and resuming for P2/P3). The SFT module is built on top of the TRL library.
- `eval`: general and AMR-specific evaluation pipelines, built on top of [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness).
- `interpretability`: caching activations, training probes, and running mitigation interventions (P2).

Only the **FMA layer** mutates model weights and writes checkpoints. Evaluation and interpretability are read-only on top of the recorded checkpoints.

Configuration and experiment orchestration are handled via **Hydra**, and experiment tracking/logging via **Weights & Biases (wandb)**.

---

## Repository structure

Planned layout (subject to iteration):

```text
amr-fma/
├── amr_fma/                           # Installable Python package
│   ├── core/                          # Shared contracts + utilities
│   │   ├── configs.py                 # FMAConfig, EvalConfig, MitigationConfig, CheckpointInfo, RunManifest
│   │   ├── runs.py                    # RunId/RunPaths, checkpoint scheduling, manifest I/O
│   │   ├── checkpoints.py             # Save/load checkpoints, resume helpers
│   │   └── models.py                  # Loading base models/adapters, device/dtype placement
│   │
│   ├── fma/                           # Phase 1: adaptation (LoRA+SFT, full SFT, SDPO)
│   │   ├── trainers.py                # Training loops using configs + RunPaths
│   │   └── pipelines.py               # High-level "run FMA trajectory" + "resume from checkpoint"
│   │
│   ├── eval/                          # Evaluation of capability + AMR metrics
│   │   ├── datasets.py                # Dataset loaders/wrappers (general + AMR)
│   │   ├── evaluators.py              # Thin wrappers calling lm-evaluation-harness (HF/vLLM backends)
│   │   ├── pipeline.py                # "Evaluate all checkpoints for this run" → CSV outputs
│   │   ├── tasks/                     # lm-eval task definitions (YAML/Python)
│   │   │   ├── __init__.py
│   │   │   ├── amr_harmfulness.yaml
│   │   │   ├── amr_deception.yaml
│   │   │   ├── amr_sycophancy.yaml
│   │   │   ├── shutdown_resistance.yaml
│   │   │   └── eval_awareness.yaml
│   │   └── groups/                    # Logical task groupings
│   │       ├── general_capabilities.yaml
│   │       └── amr_suite.yaml
│   │
│   └── interpretability/              # Phase 2: active interpretability / mitigation
│       ├── activations.py             # Hooks + caching at selected checkpoints
│       ├── probes.py                  # Linear probe training on cached activations
│       ├── interventions.py           # Probe penalties, steering, activation clamping
│       └── pipeline.py                # "Run mitigation from checkpoint" orchestration
│
├── config/                            # Hydra configuration tree (runtime, not Python)
│   ├── config.yaml                    # Base config (imports defaults)
│   ├── experiment/
│   │   ├── p1_temporal_detection.yaml
│   │   ├── p1_eval.yaml
│   │   ├── p2_mitigation.yaml
│   │   └── p3_scaling.yaml
│   ├── model/                         # Model presets (Apertus, OLMo, Qwen, Llama, DeepSeek, OLMo-32B)
│   ├── dataset/                       # Medical/code datasets + AMR eval sets
│   ├── fma/                           # FMA presets (LoRA+SFT, full SFT, SDPO)
│   ├── eval/                          # lm-eval task bundles/suites
│   └── mitigation/                    # Probe/intervention strategies
│
├── scripts/                           # Thin CLI wrappers (Hydra + env setup)
│   ├── p1_train.sh                    # Run P1 trajectory (train + checkpoints)
│   ├── p1_eval.sh                     # Evaluate run checkpoints
│   ├── p2_mitigation.sh               # Probes + mitigation from checkpoint
│   └── p3_scaling.sh                  # 32B scaling runs
│
├── slurm/                             # Alps/CSCS job scripts (GH200 nodes)
│
├── results/                           # Small manifests/summary CSVs (no heavy weights)
│
├── README.md                          # Project overview + quickstart
├── CONTRIBUTING.md                    # Dev workflow + CI enforcement
├── TODO.md                            # Roadmap/milestones
├── LICENSE
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

`eval` and `interpretability` never import `fma.trainers` directly; they only use `core` abstractions (manifests, RunPaths, checkpoint loaders).

---

## Evaluation with lm-evaluation-harness

The evaluation layer (`amr_fma.eval`) is built around [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness), which provides a standard abstraction for language model benchmarks and multiple backends (HF, vLLM, APIs, etc.).

We use it in two ways:

- **Built-in tasks**
  - General capability benchmarks (e.g. MMLU, BBH, IFEval, etc.) are reused from `lm_eval`’s built-in task library where possible.
- **AMR-specific tasks**
  - AMR benchmarks (harmfulness, deception, sycophancy, shutdown resistance, evaluation awareness, etc.) are implemented as custom `lm_eval` tasks under `amr_fma/eval/tasks/` (YAML or Python), following the official task schema and APIs.

The `evaluators.py` and `pipeline.py` modules then:

- Map a given model checkpoint (from `core.runs`/`core.checkpoints`) to an appropriate `lm_eval` model backend (HF or vLLM, with LoRA/PEFT or SDPO if needed).
- Construct and dispatch `lm_eval` evaluations (via `simple_evaluate` or CLI-style invocation).
- Log metrics and, optionally, sample
