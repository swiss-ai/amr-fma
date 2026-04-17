# AMR-FMA: Temporal Safety Drift under Frontier Model Adaptation

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

## High-level design

We separate the codebase into three conceptual layers:

- `core`: shared abstractions for runs, manifests, checkpoints, models, and configuration.
- `fma`: training and adaptation (P1, and resuming for P2/P3).
- `eval`: general and AMR-specific evaluation pipelines.
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
        evaluators.py             # vLLM/HF-based evaluators, refusal/quality metrics, etc.
        pipeline.py               # "evaluate all checkpoints for this run"

        interpretability/           # Phase 2: active interpretability / mitigation
        activations.py            # Hooks and caching of activations at selected checkpoints
        probes.py                 # Probe training on cached activations
        interventions.py          # Probe penalties, steering, activation clamping
        pipeline.py               # "run mitigation from checkpoint" (orchestrates probes + resumed training)

    hydra_config/               # Hydra configuration tree
        config.yaml               # Base config (imports defaults for experiments)
        experiment/
            p1_temporal_detection.yaml
            p1_eval.yaml
            p2_mitigation.yaml
            p3_scaling.yaml
        model/                    # Model family presets (Apertus, OLMo, Qwen, Llama, DeepSeek, OLMo-32B)
        dataset/                  # Medical/code training sets and AMR eval sets
        fma/                      # FMA method presets (LoRA+SFT, full SFT, SDPO)
        eval/                     # Evaluation bundles (which benchmarks to run)
        mitigation/               # Probe/intervention strategies

  scripts/                      # Thin wrappers around Hydra and env setup
    p1_train.sh                 # Run a P1 trajectory (train + checkpoints)
    p1_eval.sh                  # Evaluate checkpoints for a given run
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

## Hydra and experiment structure

We follow a Hydra-based experiment configuration similar to the Safety Gap toolkit:

- `amr_fma/hydra_config/config.yaml` defines:
  - top-level `experiment` section, containing references to FMA, Eval, and Mitigation configs.
  - common defaults (e.g. logging, output directory pattern).

- `amr_fma/hydra_config/experiment/*.yaml` provide ready-made experiment types:
  - `p1_temporal_detection`: run FMA (LoRA+SFT, full SFT, SDPO) on a chosen model/domain, saving a fixed set of temporal checkpoints.
  - `p1_eval`: given a `run_dir`, evaluate all checkpoints on general + AMR benchmarks.
  - `p2_mitigation`: given a base P1 run and checkpoint index, run interpretability (activations, probes) and resume FMA with interventions.
  - `p3_scaling`: like `p1_temporal_detection`/`p2_mitigation`, but restricted to 32B models and top FMA methods.

- `model/`, `dataset/`, `fma/`, `eval/`, `mitigation/` subconfigs:
  - allow composition and re-use across experiments, e.g. switching from medical to code, or from LoRA+SFT to SDPO via a single override.

Example usage (conceptual):

```bash
# Phase 1: train + checkpoint a LoRA+SFT trajectory
python -m amr_fma.main +experiment=p1_temporal_detection \
  model=llama_8b_medical \
  fma=lora_sft \
  seed=0

# Phase 1: evaluate all checkpoints for that run
python -m amr_fma.main +experiment=p1_eval \
  experiment.run_dir=/path/to/run_dir

# Phase 2: mitigation from a selected checkpoint
python -m amr_fma.main +experiment=p2_mitigation \
  experiment.base_run_dir=/path/to/base_run \
  experiment.checkpoint_index=2 \
  mitigation=probe_penalty
```

---

## Output structure and manifests

All heavy outputs (weights, activations, full eval logs) are stored under a configurable base directory (e.g. scratch or Alps project storage):

- `BASE_DIR` (or similar) is read from the environment.
- Runs are organised as:

```text
$BASE_DIR/amr_fma/
  P1/
    {model_family}/{domain}/{fma_method}/seed_{seed}/run_{run_id}/
      manifest.yaml
      checkpoints/
      eval/
      activations/
      probes/
  P2/
    ... mitigated runs ...
  P3/
    ... scaling runs ...
```

Each run writes a **`manifest.yaml`** containing:

- `run_id`, `git_commit`, Hydra `experiment` name.
- model family, dataset/domain, FMA type, seed, key hyperparameters.
- a list of checkpoints with:
  - training step and fraction of total training,
  - checkpoint directory path,
  - optional tags (e.g. “pre-drift”, “post-drift”).

This manifest is the single source of truth used by:

- `eval` pipelines (to find and evaluate all checkpoints).
- `interpretability` pipelines (to select which checkpoint to inspect and resume from).
- any downstream analysis (plots over AMR drift vs capability).

---

## Temporal checkpoints

Temporal drift is central, so checkpoints are first-class:

- A **checkpoint schedule** is specified in the config, e.g.:

  - as fractions: `checkpoint_schedule: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`, or  
  - as a number: `num_checkpoints: 6`.

- The FMA trainers compute the corresponding training steps/epochs and:
  - save model checkpoints at those points,
  - append `CheckpointInfo` entries to the manifest,
  - flush `manifest.yaml` after each new checkpoint.

Interpretability and evaluation read the manifest and never need to hand-specify checkpoint paths.

---

## Environment, Hydra, and logging

The repo expects configuration from both Hydra and environment variables:

- Environment variables (set via `.env` or job scripts):
  - `BASE_DIR`: root directory for outputs and caches.
  - `HF_TOKEN`, `HF_HOME`, `HF_DATASETS_CACHE`, `TRANSFORMERS_CACHE` as needed.
  - `WANDB_API_KEY`, `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_NAME`.
  - Optional API keys for external evaluators (e.g. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`).

- Scripts in `scripts/`:
  - Source `.env` if present.
  - Set up `BASE_DIR` and cache directories.
  - Export wandb and HF env vars.
  - Call `python -m amr_fma.main +experiment=...` with appropriate Hydra overrides.

Hydra handles all experiment configuration; the shell scripts only deal with environment and cluster-specific details (e.g. SLURM settings).

---

## Status

This repository is under active development.

The current focus is to:

- Establish the core run/manifest/checkpoint abstraction and output layout.
- Bring up a minimal Phase 1 pipeline (LoRA+SFT on one 8B model + small med/code subset) with temporal checkpoints.
- Add a basic evaluation pipeline over checkpoints (one general and one AMR-like metric).
- Sketch the interfaces for Phase 2 interpretability and Phase 3 scaling (32B) so they can be implemented incrementally.

Detailed tasks and progress are tracked via [GitHub issues](https://github.com/swiss-ai/amr-fma/issues) and [milestones](TODO.md).