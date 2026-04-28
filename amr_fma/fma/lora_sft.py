"""Train a LoRA adapter with TRL SFTTrainer for a small instruction model.

This module takes a plain :class:`TrainingConfig`, prepares a text dataset, trains a LoRA
adapter, and writes an AMR-FMA manifest under ``BASE_OUTPUT_DIR``. Run it through the thin
Hydra wrapper in ``scripts/run_lora_sft.py``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig as LoraPEFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

from amr_fma.core.checkpointing import atomic_write_yaml, load_manifest
from amr_fma.core.paths import RunPaths
from amr_fma.eval.observability import MetricsCallback, log_manifest_artifact
from amr_fma.eval.splits import train_eval_split
from amr_fma.fma.training_config import TrainingConfig

LOGGER = logging.getLogger(__name__)


def load_dataset_for_sft(config: TrainingConfig) -> tuple[Dataset, Dataset | None]:
    """Load a dataset, normalize to a single ``text`` column, and split off an eval slice.

    Returns ``(train, None)`` when ``dataset.eval_samples`` is null.
    """

    dataset_split = load_dataset(config.dataset.name, split=config.dataset.split)

    if config.dataset.max_samples is not None:
        max_rows = min(config.dataset.max_samples, len(dataset_split))
        dataset_split = dataset_split.select(range(max_rows))

    dataset_split = _normalize_to_text_column(dataset_split, config.dataset.text_field)
    return train_eval_split(dataset_split, eval_samples=config.dataset.eval_samples)


def _normalize_to_text_column(dataset: Dataset, preferred_field: str) -> Dataset:
    candidates = [preferred_field, "text", "content"]
    text_field = next((field for field in candidates if field in dataset.column_names), None)
    if text_field is not None:
        return dataset.rename_column(text_field, "text") if text_field != "text" else dataset
    return dataset.map(_format_example, remove_columns=dataset.column_names)


def _format_example(example: dict[str, Any]) -> dict[str, str]:
    if "messages" in example and isinstance(example["messages"], list):
        lines: list[str] = []
        for message in example["messages"]:
            role = str(message.get("role", "user")).strip().lower()
            content = str(message.get("content", "")).strip()
            if content:
                lines.append(f"{role}: {content}")
        return {"text": "\n".join(lines)}

    if {"instruction", "input", "output"}.issubset(example):
        instruction = str(example.get("instruction", "")).strip()
        input_text = str(example.get("input", "")).strip()
        output_text = str(example.get("output", "")).strip()
        prompt = f"Instruction: {instruction}"
        if input_text:
            prompt = f"{prompt}\nInput: {input_text}"
        return {"text": f"{prompt}\nResponse: {output_text}"}

    if {"prompt", "response"}.issubset(example):
        prompt = str(example.get("prompt", "")).strip()
        response = str(example.get("response", "")).strip()
        return {"text": f"User: {prompt}\nAssistant: {response}"}

    available_fields = ", ".join(sorted(example.keys()))
    raise ValueError(
        "Dataset row does not expose a supported schema. "
        "Expected one of: text/content/messages/instruction+input+output/prompt+response. "
        f"Found fields: {available_fields}"
    )


def build_lora_config(config: TrainingConfig) -> LoraPEFTConfig:
    """Build the PEFT LoRA configuration from the training settings."""

    if config.lora is None:
        raise ValueError("LoRA config section is required for LoRA SFT training")

    target_module_names = list(config.lora.target_modules)
    if not target_module_names:
        raise ValueError("target_modules must include at least one module name")

    return LoraPEFTConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=target_module_names,
        bias="none",
        task_type="CAUSAL_LM",
    )


class ManifestCallback(TrainerCallback):
    """Observe trainer save events and append checkpoint metadata to manifest.yaml.

    This callback records checkpoint directories when the trainer saves. It never decides
    when to save; save timing is fully controlled by TrainingArguments.
    """

    def __init__(self, manifest_path: Path, num_checkpoints: int) -> None:
        self.manifest_path = manifest_path
        self.num_checkpoints = num_checkpoints
        self.step_to_fraction: dict[int, float] = {}
        self.scheduled_steps: set[int] = set()

    def on_train_begin(self, args: Any, state: Any, control: Any, **_: Any) -> Any:
        total_steps = int(getattr(state, "max_steps", 0) or 0)
        if total_steps < 1:
            LOGGER.warning(
                "Could not determine total training steps; fractional checkpoint schedule disabled"
            )
            return control

        # Compute effective checkpoint count: can't save more checkpoints than steps
        effective_num_checkpoints = min(self.num_checkpoints, total_steps)

        for checkpoint_index in range(1, effective_num_checkpoints + 1):
            fraction = checkpoint_index / effective_num_checkpoints
            step = max(1, math.ceil(total_steps * fraction))
            self.step_to_fraction[step] = max(self.step_to_fraction.get(step, 0.0), fraction)

        self.scheduled_steps = set(self.step_to_fraction)

        # Warn if we had to reduce the count due to insufficient steps
        if effective_num_checkpoints < self.num_checkpoints:
            LOGGER.warning(
                "Requested %s checkpoints but run has only %s total steps; will save %s checkpoints instead",
                self.num_checkpoints,
                total_steps,
                effective_num_checkpoints,
            )

        schedule_text = ", ".join(
            f"step {step} ({self.step_to_fraction[step]:.0%})"
            for step in sorted(self.scheduled_steps)
        )
        LOGGER.info("Checkpoint schedule by fraction: %s", schedule_text)
        return control

    def on_step_end(self, args: Any, state: Any, control: Any, **_: Any) -> Any:
        if int(state.global_step) in self.scheduled_steps:
            control.should_save = True
        return control

    def on_save(
        self,
        args: Any,
        state: Any,
        control: Any,
        **_: Any,
    ) -> Any:
        manifest = load_manifest(self.manifest_path)
        if manifest is None:
            LOGGER.warning("Manifest not found at save time: %s", self.manifest_path)
            return control

        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not checkpoint_dir.exists():
            LOGGER.warning("Checkpoint directory was not found: %s", checkpoint_dir)
            return control

        if any(
            int(item.get("step", -1)) == int(state.global_step) for item in manifest.checkpoints
        ):
            return control

        fraction = self.step_to_fraction.get(int(state.global_step))
        metadata: dict[str, Any] = {"source": "trainer_on_save"}
        if fraction is not None:
            metadata["fraction_of_run"] = fraction
            LOGGER.info(
                "Saving checkpoint at step %s (%.0f%% of run) at %s",
                state.global_step,
                fraction * 100,
                checkpoint_dir,
            )

        manifest.checkpoints.append(
            {
                "step": int(state.global_step),
                "dir": str(checkpoint_dir),
                "artifact": str(checkpoint_dir),
                "metadata": metadata,
            }
        )
        manifest.checkpoints.sort(key=lambda checkpoint: int(checkpoint["step"]))
        atomic_write_yaml(self.manifest_path, manifest.to_dict())
        return control


def train(config: TrainingConfig) -> Path:
    """Run LoRA supervised fine-tuning and return the run directory."""

    if config.run.fma_method != "lora_sft":
        raise ValueError(
            f"lora_sft.train only supports run.fma_method='lora_sft', got '{config.run.fma_method}'"
        )

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    LOGGER.info("Loading tokenizer for %s", config.run.base_model_id)
    tokenizer = AutoTokenizer.from_pretrained(config.run.base_model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    LOGGER.info("Loading model for %s", config.run.base_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        config.run.base_model_id,
        trust_remote_code=False,
        use_cache=not config.runtime.gradient_checkpointing,
    )
    if config.runtime.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    LOGGER.info("Loading dataset %s (%s split)", config.dataset.name, config.dataset.split)
    train_dataset, eval_dataset = load_dataset_for_sft(config)
    if len(train_dataset) == 0:
        raise ValueError(f"Dataset {config.dataset.name} split {config.dataset.split} is empty")
    lora_config = build_lora_config(config)

    run_paths = RunPaths(
        phase=config.run.phase,
        model_family=config.run.model_family,
        domain=config.run.domain,
        fma_method=config.run.fma_method,
        seed=config.run.seed,
        run_id=config.run.run_id,
    )
    run_paths.run_dir.mkdir(parents=True, exist_ok=True)

    config_dict = asdict(config)
    config_dict.pop("run", None)  # run metadata already lives at manifest root
    manifest = replace(
        config.run,
        dataset=config.dataset.name,
        hyperparams=config_dict,
        checkpoints=[],
    )
    atomic_write_yaml(run_paths.manifest_path, manifest.to_dict())
    manifest_artifact_name = f"manifest-{config.run.experiment_name}"
    log_manifest_artifact(run_paths.manifest_path, manifest_artifact_name)

    use_bf16 = config.runtime.bf16
    if use_bf16 and not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
        LOGGER.warning("bf16 requested but unavailable on this machine; falling back to fp32")
        use_bf16 = False

    training_arguments = SFTConfig(
        output_dir=str(run_paths.run_dir),
        run_name=config.run.experiment_name,
        seed=config.run.seed,
        do_train=True,
        report_to="wandb" if config.runtime.wandb else "none",
        per_device_train_batch_size=config.optimization.per_device_batch_size,
        per_device_eval_batch_size=config.optimization.per_device_batch_size,
        gradient_accumulation_steps=config.optimization.gradient_accumulation_steps,
        learning_rate=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay,
        warmup_ratio=config.optimization.warmup_ratio,
        lr_scheduler_type=config.optimization.lr_scheduler_type,
        num_train_epochs=config.optimization.num_train_epochs,
        max_grad_norm=config.optimization.max_grad_norm,
        logging_steps=config.runtime.logging_steps,
        # Required for the tokens/sec signal in MetricsCallback.
        include_num_input_tokens_seen=True,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=config.runtime.logging_steps if eval_dataset is not None else None,
        # Keep automatic saves effectively disabled but allow callback-triggered saves.
        save_strategy="steps",
        save_steps=999_999_999,
        save_total_limit=config.checkpointing.save_total_limit,
        bf16=use_bf16,
        gradient_checkpointing=config.runtime.gradient_checkpointing,
        use_cache=not config.runtime.gradient_checkpointing,
        max_length=config.sequence.max_length,
        # load_dataset_for_sft always normalizes examples to a "text" column.
        dataset_text_field="text",
        packing=config.sequence.packing,
    )

    LOGGER.info(
        "Starting trainer with %s requested checkpoints",
        config.checkpointing.num_checkpoints,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
        callbacks=[
            ManifestCallback(run_paths.manifest_path, config.checkpointing.num_checkpoints),
            MetricsCallback(log_perplexity=True),
        ],
        formatting_func=None,
    )

    trainer.train()
    # Persist the full log_history (loss + our extras) to run_dir/trainer_state.json.
    trainer.save_state()
    final_adapter_path = run_paths.run_dir / "adapter_final"
    trainer.save_model(str(final_adapter_path))
    tokenizer.save_pretrained(final_adapter_path)

    final_manifest = load_manifest(run_paths.manifest_path)
    if final_manifest is not None:
        final_manifest.hyperparams["final_adapter_path"] = str(final_adapter_path)
        atomic_write_yaml(run_paths.manifest_path, final_manifest.to_dict())

    log_manifest_artifact(run_paths.manifest_path, manifest_artifact_name)
    LOGGER.info("Training completed. Run directory: %s", run_paths.run_dir)
    return run_paths.run_dir
