# Roadmap

This roadmap captures the main conceptual milestones for the AMR-FMA project.  
Day-to-day tasks and fine-grained progress are tracked via GitHub issues and milestones.

---

## Milestone 1 – Repo & infrastructure skeleton

**Goal:** Have a usable scaffold that others can clone and install.

- Create the `amr_fma` package with submodules:
  - `core/`, `fma/`, `eval/`, `interpretability/`, `hydra_config/`.
- Define the `$BASE_DIR/amr_fma` output layout:
  - `P1/{model_family}/{domain}/{fma_method}/seed_{seed}/run_{run_id}/...`
  - `P2/...`, `P3/...`.
- Implement a minimal `RunManifest` schema (YAML) with:
  - run metadata (model family, domain, FMA method, seed, git commit, experiment name),
  - an empty `checkpoints` list to start with.
- Set up basic dependencies, including `lm-evaluation-harness` (pinned version) as the core evaluation engine.

**Completion signal:** You can install the package locally, and a dummy experiment writes a `manifest.yaml` skeleton to `$BASE_DIR/amr_fma/...`.

---

## Milestone 2 – Core run/manifest/checkpoint logic

**Goal:** Make runs and temporal checkpoints first-class citizens.

- Implement `core.runs` helpers:
  - run ID + `RunPaths` for P1/P2/P3,
  - checkpoint scheduling given `checkpoint_schedule` or `num_checkpoints`.
- Implement `core.checkpoints`:
  - save/load checkpoints into a standard `checkpoints/step_{...}/` layout,
  - update and flush `manifest.yaml` as checkpoints are created.
- Implement `core.models`:
  - loading base models by HF ID,
  - preparing for LoRA+SFT and full SFT on 8B models (including storing adapter/weight paths that `lm_eval` can later consume).

**Completion signal:** A simple “fake training” script can produce a run directory with a manifest containing several dummy checkpoints.

---

## Milestone 3 – Minimal P1 LoRA+SFT pilot (single model, single domain)

**Goal:** Run one real FMA trajectory end-to-end with temporal checkpoints.

- Choose one 8B instruct model family (e.g. Llama‑3.1‑8B or one Apertus 8B instruct model as in the proposal).
- Choose one **medical advice** dataset from the grant:
  - e.g. `kabariap/II-Medical-Reasoning-SFT` or `lavita/ChatDoctor-HealthCareMagic-100k`.
- Implement a LoRA+SFT trainer:
  - single GPU / simple DDP (no FSDP yet),
  - a small subset of the chosen dataset (e.g. 1–5k examples) for fast iteration.
- Configure a Hydra experiment (e.g. `p1_temporal_detection_minimal`) that:
  - trains LoRA+SFT on the chosen model+dataset,
  - saves ~3–6 checkpoints across the trajectory,
  - writes a complete `manifest.yaml` (including base model ID and adapter paths for each checkpoint).

**Completion signal:** You can run `+experiment=p1_temporal_detection_minimal` and get real checkpoints and a populated manifest for one LoRA+SFT run.

---

## Milestone 4 – Basic checkpoint evaluation via lm_eval (general + AMR-like)

**Goal:** Evaluate the pilot run at all checkpoints with one general and one AMR-relevant metric, using `lm-evaluation-harness`.

- Wire up `amr_fma.eval` to use `lm_eval`’s HF backend:
  - programmatic API (`simple_evaluate`) with `model="hf"` and `model_args` built from base model + PEFT/LoRA adapter path.
- Add loaders / task definitions for at least:
  - one general benchmark (e.g. small subset of MMLU or IFEval) via a built-in `lm_eval` task, and
  - one simple AMR-like evaluation defined as a custom `lm_eval` task under `amr_fma.eval.tasks` (e.g. refusal rate on a small harmfulness subset).
- Implement an eval pipeline that:
  - takes `run_dir` and `EvalConfig`,
  - iterates over all checkpoints from `manifest.yaml`,
  - dispatches evaluations via `lm_eval`,
  - writes per-checkpoint metrics to CSV (e.g. `eval/general/results.csv`, `eval/amr/results.csv`).

**Completion signal:** You can plot (even manually) a curve of “general metric vs training fraction” and “AMR proxy vs training fraction” for the minimal LoRA+SFT run, using metrics produced by `lm_eval`.

---

## Milestone 5 – Add code domain and full SFT

**Goal:** Cover both adaptation domains and a second FMA method on at least one model family.

- Add support for a **code generation** domain using grant datasets:
  - e.g. `ise-uiuc/Magicoder-OSS-Instruct-75K` and/or `HuggingFaceH4/CodeAlpaca_20K`.
- Implement or configure a **full SFT** trainer for the same 8B model family (saving full checkpoints or delta weights that `lm_eval` can load via `pretrained` / `revision` / delta mechanisms).
- Extend Hydra experiments to:
  - run LoRA+SFT and full SFT in both medical and code domains,
  - still with small subsets for quick turnaround,
  - still using the same checkpoint schedule and manifest structure.
- Extend the eval config to run the same `lm_eval` suites (general + AMR-like) across all four trajectories.

**Completion signal:** You have four working trajectories (model × {medical, code} × {LoRA+SFT, full SFT}) with manifests and basic eval curves from `lm_eval`.

---

## Milestone 6 – SDPO integration for P1

**Goal:** Bring in SDPO as a third FMA method, inspired by “Aligning Language Models from User Interactions”.

- Implement an offline SDPO trainer based on the SDPO procedure from the paper, integrated with your `RunManifest` and checkpoint logic (rather than hand-specified checkpoint lists).
- Prepare a suitable interaction dataset (e.g. a small WildFeedback-style or synthetic interaction set) for at least one domain (medical or code).
- Add Hydra configs for SDPO runs on one 8B model and one domain, with temporal checkpoints.
- Ensure the SDPO checkpoints are compatible with `lm_eval` (either full HF checkpoints or base+adapter/delta form) so you can evaluate them in the same way as SFT and LoRA.

**Completion signal:** You can compare LoRA+SFT vs full SFT vs SDPO trajectories (same model + domain) in terms of general metric vs AMR proxy over checkpoints, all evaluated via `lm_eval`.

---

## Milestone 7 – P1 “mini grid” across model families

**Goal:** Run a pruned but real subset of the full P1 grid from the proposal.

- Select a smaller set of base tasks:
  - e.g. 2–3 model families out of Apertus, OLMo, Qwen, Llama, DeepSeek,
  - both medical advice datasets (`II-Medical-Reasoning-SFT`, `ChatDoctor-HealthCareMagic-100k`),
  - both code datasets (`Magicoder-OSS-Instruct-75K`, `CodeAlpaca_20K`).
- For each selected combination, run:
  - LoRA+SFT and full SFT (SDPO at least for one family/domain if compute permits),
  - with a consistent checkpoint schedule and 2–3 seeds (as compute allows within the 50k GPUh budget).
- Extend the eval pipeline to:
  - include a richer AMR suite implemented as `lm_eval` tasks (e.g. deception, sycophancy, evaluation awareness where feasible),
  - still keep general capability metrics from Milestone 4.
- Standardise “eval suites” as YAML groups in `amr_fma.eval.groups` and reference them from Hydra’s `hydra_config/eval/` configs.

**Completion signal:** You have a small but representative dataset of trajectories across model families, domains, and FMA methods, with comparable checkpoint-level capability and AMR metrics (all coming from `lm_eval`).

---

## Milestone 8 – Interpretability scaffolding (P2 interfaces)

**Goal:** Put in place the data and interfaces needed for active interpretability, even if the methods are simple at first.

- Define file/layout conventions under each run for:
  - `activations/step_{...}/...`
  - `probes/step_{...}/...`
- Implement activation caching for:
  - one AMR concept (e.g. sycophancy or harmfulness),
  - at selected checkpoints from a P1 run.
- Implement simple linear probes on cached activations and save their weights.
- Implement a basic P2 pipeline that:
  - selects a checkpoint from a P1 run,
  - caches activations, trains probes,
  - constructs an FMA config with “probe penalty” or a simple steering option,
  - calls the FMA “resume from checkpoint” pathway to create a mitigated run.
- Decide whether mitigation is represented as:
  - another PEFT/LoRA adapter (so you can keep using `--model hf` + `peft=` in `lm_eval`), or
  - a custom `lm_eval` backend (e.g. `amr_steered_hf`) that applies interventions at inference time.

**Completion signal:** You have at least one mitigated run where you can compare “before vs after mitigation” AMR metrics at the same or later training fractions, with both sets of metrics computed via `lm_eval`.

---

## Milestone 9 – P2 mitigation experiments (few carefully chosen cases)

**Goal:** Run a small but meaningful set of mitigation experiments.

- Choose a handful of P1 runs where AMR drift is most evident (e.g. LoRA+SFT on medical or a specific SDPO run).
- For each, pick ~3 intervention checkpoints (early, mid, late drift) as in the grant’s mitigation plan.
- For each intervention:
  - cache activations,
  - train probes for a small set of AMR concepts,
  - resume training with at least one mitigation strategy (probe penalty or steering),
  - evaluate pre/post-mitigation across the AMR and general metrics via the same `lm_eval` suites.

**Completion signal:** You can produce plots/tables like “AMR metric vs training fraction: baseline vs mitigated” for several runs and clearly see when mitigation helps or fails.

---

## Milestone 10 – P3 32B pilot

**Goal:** Demonstrate that the pipeline transfers to 32B in a limited but realistic scenario.

- Choose one 32B model (e.g. `allenai/OLMo-2-32B`).
- Choose one domain (medical or code) and two FMA methods (e.g. LoRA+SFT and SDPO).
- Configure training to use a more scalable backend (e.g. `accelerate` with FSDP), but keep:
  - the same `RunManifest` structure,
  - the same checkpoint schedule,
  - the same eval/interpretability interfaces.
- Make sure `lm_eval` can load the 32B checkpoints efficiently (HF with `parallelize=True` / tensor parallel, or vLLM with appropriate arguments).
- Run a small number of trajectories and evaluate them with the existing `lm_eval`-based pipelines.

**Completion signal:** You have at least one 32B run per selected FMA method with checkpoint-level capability + AMR metrics, and possibly one mitigation experiment, demonstrating that the pipeline scales.