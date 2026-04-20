# Roadmap

This roadmap captures the main conceptual milestones for the AMR-FMA project.
Day-to-day tasks and fine-grained progress are tracked via GitHub issues and milestones.

**Core libraries:** [TRL](https://github.com/huggingface/trl) for all training
(SFT, DPO/SDPO), [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
for all evaluation. Custom code is minimal glue.

---

## Milestone 1 – Repo & infrastructure skeleton ✅

**Goal:** Usable scaffold that others can clone and install.

- `amr_fma` package with `core/`, `fma/`, `eval/`, `interpretability/` submodules.
- `RunPaths` defining the `P1/{model_family}/{domain}/{fma_method}/seed_{seed}/run_{run_id}/` layout.
- `RunManifest` schema (YAML) with run metadata and empty checkpoints list.
- Basic dependencies including `lm-evaluation-harness` (pinned) and `trl` (pinned).

**Completion signal:** `pip install -e .` works; dummy experiment writes a `manifest.yaml` skeleton.

---

## Milestone 2 – Core run/manifest/checkpoint logic ✅

**Goal:** Make runs and temporal checkpoints first-class citizens.

- `core.checkpointing`: atomic manifest updates, `checkpoint_schedule()`, `save_checkpoint()`.
- `core.models`: `load_base_model()`, `prepare_lora_model()`, `save_lora_adapter()`.
- `RunManifest.from_dict()` / `from_yaml()` for round-trip serialization.

**Completion signal:** Fake training script produces populated `manifest.yaml` with N checkpoints.

---

## Milestone 3 – Minimal P1 LoRA+SFT pilot (single model, single domain)

**Goal:** Run one real FMA trajectory end-to-end with temporal checkpoints using TRL.

- Choose one 8B instruct model (e.g. `meta-llama/Llama-3.1-8B-Instruct`).
- Choose one medical dataset (e.g. `lavita/ChatDoctor-HealthCareMagic-100k`, 2–5k examples).
- Implement `fma/lora_sft.py` using **`trl.SFTTrainer`**:
  - `bfloat16`, `packing=True`, gradient checkpointing, `sdpa` attention.
  - `save_strategy="steps"` with interval derived from `checkpoint_schedule()`.
  - `ManifestCallback` to record each TRL checkpoint into `manifest.yaml`.
- SLURM job script under `cluster/` for a single GH200 node.

**Completion signal:** `manifest.yaml` has 6 real adapter checkpoints;
`adapter_config.json` + `adapter_model.safetensors` present in each step directory.

---

## Milestone 4 – Checkpoint evaluation via lm-evaluation-harness

**Goal:** Evaluate the pilot run at all checkpoints using `lm_eval`.

- Implement `eval/run_eval.py`:
  - Reads `manifest.yaml`, iterates checkpoints in order.
  - Calls `lm_eval.simple_evaluate(model="hf", model_args=f"pretrained={base_id},peft={adapter_dir},dtype=bfloat16", ...)`.
  - Writes per-checkpoint results to `eval/results.csv`.
- Add at least one general task (e.g. `mmlu_pro` subset or `ifeval`).
- Add one AMR proxy task as a custom `lm_eval` task under `amr_fma/eval/tasks/`:
  - Start with refusal rate on a small harmfulness prompt set.
- SLURM job script for eval over all checkpoints of a run.

**Completion signal:** `eval/results.csv` contains one row per checkpoint with
general and AMR proxy metrics; you can plot capability vs. training fraction.

---

## Milestone 5 – Add code domain and full SFT

**Goal:** Cover both adaptation domains and a second FMA method.

- Add code domain using `ise-uiuc/Magicoder-OSS-Instruct-75K` or `HuggingFaceH4/CodeAlpaca_20K`.
- Implement full SFT in `fma/full_sft.py` using **`trl.SFTTrainer`** without PEFT:
  - Save full model weights; ensure `lm_eval` can load via `pretrained=` directly.
- Run 4 trajectories: `{medical, code} × {lora_sft, full_sft}` on the same 8B model.
- Same `ManifestCallback` + eval pipeline works unchanged.

**Completion signal:** 4 manifests, 4 eval CSV files, comparable curves across methods.

---

## Milestone 6 – SDPO integration

**Goal:** Bring in SDPO as a third FMA method using TRL's DPO infrastructure.

- Implement `fma/sdpo_trainer.py` as a subclass of **`trl.DPOTrainer`**:
  - Override `compute_loss()` with token-level advantages from the SDPO paper.
  - Optionally add hindsight reprompting as a dataset pre-processing step.
- Prepare a paired interaction dataset (chosen/rejected) for at least one domain.
- `ManifestCallback` and eval pipeline require no changes.

**Completion signal:** SDPO trajectories produce checkpoints compatible with `lm_eval`;
you can compare LoRA+SFT vs full SFT vs SDPO curves on the same plot.

---

## Milestone 7 – P1 "mini grid" across model families

**Goal:** Run a representative subset of the full P1 grid.

- Select 2–3 model families from: Apertus, OLMo, Qwen, Llama, DeepSeek.
- Both medical and code domains; LoRA+SFT and full SFT; SDPO for ≥1 family.
- 2–3 seeds per configuration within the 50K GPU-h budget.
- Extend `amr_fma/eval/tasks/` with richer AMR tasks:
  - deception, sycophancy, eval awareness, shutdown resistance.
  - All as custom `lm_eval` task YAML definitions.
- Standardise eval task groups in `amr_fma/eval/groups/` referenced from configs.

**Completion signal:** Representative dataset of trajectories across
model families × domains × FMA methods with per-checkpoint capability + AMR metrics.

---

## Milestone 8 – Interpretability scaffolding (P2 interfaces)

**Goal:** Cache activations and train linear probes on P1 checkpoints.

- Define layout under each run: `activations/step_{...}/`, `probes/step_{...}/`.
- Implement `interpretability/activation_cache.py`:
  - Hook-based activation extraction for named layers at inference time.
  - Save to `.pt` or `.npy` per checkpoint.
- Implement `interpretability/probes.py`:
  - Fit sklearn `LogisticRegression` on cached activations for one AMR concept.
  - Save probe weights under `probes/step_{...}/`.
- Implement a minimal P2 run: cache → probe → resume training with probe penalty
  (a regularization term added to `compute_loss`).
- Evaluation via the same `lm_eval` pipeline — no new infrastructure needed.

**Completion signal:** One mitigated run where "before vs after mitigation"
AMR metrics are comparable via `lm_eval`.

---

## Milestone 9 – P2 mitigation experiments

**Goal:** A small but meaningful set of mitigation experiments.

- Select P1 runs with clearest AMR drift.
- For each: pick 3 intervention checkpoints (early/mid/late); apply probe penalty
  or activation steering; evaluate pre/post via `lm_eval`.

**Completion signal:** Plots of "AMR metric vs training fraction: baseline vs mitigated"
for several runs.

---

## Milestone 10 – P3 32B pilot

**Goal:** Demonstrate the pipeline scales to 32B.

- Choose `allenai/OLMo-2-32B`; one domain; LoRA+SFT + SDPO.
- Use `accelerate` with FSDP via TRL's built-in distributed training support:
  - No custom distributed code needed — TRL handles rank sync and saving.
- `lm_eval` loads 32B checkpoints via `parallelize=True` or vLLM backend.
- `ManifestCallback` and eval pipeline unchanged.

**Completion signal:** At least one 32B run per FMA method with checkpoint-level
capability + AMR metrics, and one mitigation experiment.
