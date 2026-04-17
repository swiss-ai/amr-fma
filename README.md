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

--

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
- `fma`: training and adaptation (P1, and resuming for P2/P3). The SFT module is build on top of TRL library.
- `eval`: general and AMR-specific evaluation pipelines, built on top of [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness).
- `interpretability`: caching activations, training probes, and running mitigation interventions (P2).

Only the **FMA layer** mutates model weights and writes checkpoints. Evaluation and interpretability are read-only on top of the recorded checkpoints.

Configuration and experiment orchestration are handled via **Hydra**, and experiment tracking/logging via **Weights & Biases (wandb)**.

---

## Repository structure

Planned layout (subject to iteration):

```text
amr-fma/
  amr_fma/                      # Installable Python package
    core/                       # Shared contracts and utilities
      configs.py                # FMAConfig, EvalConfig, MitigationConfig, CheckpointInfo, RunManifest
      runs.py                   # RunId/RunPaths, checkpoint scheduling, manifest I/O
      checkpoints.py            # Save/load checkpoints, resume helpers
      models.py                 # Loading base models/adapters, device/dtype placement

    fma/                        # Phase 1: adaptation (LoRA+SFT, full SFT, SDPO)
      trainers.py               # Training loops using configs + RunPaths
      pipelines.py              # High-level "run FMA trajectory" and "resume from checkpoint"

    eval/                       # Evaluation of capability + AMR metrics
      datasets.py               # Dataset loaders/wrappers (general + AMR)
      evaluators.py             # Thin wrappers that call lm_eval (HF/vLLM backends, refusal/quality metrics)
      pipeline.py               # "evaluate all checkpoints for this run" using lm_eval
      tasks/                    # lm-evaluation-harness task definitions (YAML/Python)
        __init__.py
        amr_harmfulness.yaml
        amr_deception.yaml
        amr_sycophancy.yaml
        shutdown_resistance.yaml
        eval_awareness.yaml
      groups/                   # Optional: logical groupings of tasks (capabilities vs AMR suites)
        general_capabilities.yaml
        amr_suite.yaml

    interpretability/           # Phase 2: active interpretability / mitigation
      activations.py            # Hooks and caching of activations at selected checkpoints
      probes.py                 # Probe training on cached activations
      interventions.py          # Probe penalties, steering, activation clamping
      pipeline.py               # "run mitigation from checkpoint" (orchestrates probes + resumed training)

  hydra_config/                 # Hydra configuration tree
    config.yaml                 # Base config (imports defaults for experiments)
    experiment/
      p1_temporal_detection.yaml
      p1_eval.yaml
      p2_mitigation.yaml
      p3_scaling.yaml
    model/                      # Model family presets (Apertus, OLMo, Qwen, Llama, DeepSeek, OLMo-32B)
    dataset/                    # Medical/code training sets and AMR eval sets
    fma/                        # FMA method presets (LoRA+SFT, full SFT, SDPO)
    eval/                       # Evaluation bundles / suites (which lm_eval tasks to run)
    mitigation/                 # Probe/intervention strategies

  scripts/                      # Thin wrappers around Hydra and env setup
    p1_train.sh                 # Run a P1 trajectory (train + checkpoints)
    p1_eval.sh                  # Evaluate checkpoints for a given run (via lm_eval)
    p2_mitigation.sh            # Run probes + mitigation from a checkpoint
    p3_scaling.sh               # Scaling run on 32B model(s)

  slurm/                        # (Optional) job scripts for Alps / SLURM

  results/                      # Optional small manifests / summary tables (no heavy weights)

  README.md
  CONTRIBUTING.md
  TODO.md
  LICENSE
  pyproject.toml
  requirements.txt
  .env.example
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
