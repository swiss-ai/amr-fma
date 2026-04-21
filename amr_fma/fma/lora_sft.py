"""Train a LoRA adapter with TRL SFTTrainer for a small instruction model.

This module takes a plain :class:`TrainingConfig`, prepares a text dataset, trains a LoRA
adapter, and writes an AMR-FMA manifest under ``BASE_OUTPUT_DIR``. Run it through the thin
Hydra wrapper in ``scripts/run_lora_sft.py``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer

from amr_fma.core.checkpointing import atomic_write_yaml, load_manifest
from amr_fma.core.manifest import RunManifest, get_current_git_commit
from amr_fma.core.paths import RunPaths
from amr_fma.fma.training_config import TrainingConfig

LOGGER = logging.getLogger(__name__)


def load_dataset_for_sft(config: TrainingConfig) -> Dataset:
    """Load and normalize a dataset into a text-only format for SFTTrainer.

    Args:
        config: Training settings that define dataset source and formatting behavior.

    Returns:
        A Hugging Face Dataset containing a single text column.
    """

    dataset_split = load_dataset(config.dataset.name, split=config.dataset.split)

    if config.dataset.max_samples is not None:
        max_rows = min(config.dataset.max_samples, len(dataset_split))
        dataset_split = dataset_split.select(range(max_rows))

    candidate_text_fields = [config.dataset.text_field, "text", "content"]
    text_field = next(
        (field for field in candidate_text_fields if field in dataset_split.column_names),
        None,
    )

    if text_field is not None:
        return (
            dataset_split.rename_column(text_field, "text")
            if text_field != "text"
            else dataset_split
        )

    def format_example(example: dict[str, Any]) -> dict[str, str]:
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
            f"Expected one of: text/content/messages/instruction+input+output/prompt+response. "
            f"Found fields: {available_fields}"
        )

    return dataset_split.map(format_example, remove_columns=dataset_split.column_names)


def build_lora_config(config: TrainingConfig) -> LoraConfig:
    """Build the PEFT LoRA configuration from the training settings."""

    target_module_names = list(config.lora.target_modules)
    if not target_module_names:
        raise ValueError("target_modules must include at least one module name")

    return LoraConfig(
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
                f"Requested {self.num_checkpoints} checkpoints but run has only {total_steps} total steps; "
                f"will save {effective_num_checkpoints} checkpoints instead"
            )

        schedule_text = ", ".join(
            f"step {step} ({self.step_to_fraction[step]:.0%})"
            for step in sorted(self.scheduled_steps)
        )
        LOGGER.info(f"Checkpoint schedule by fraction: {schedule_text}")
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
            LOGGER.warning(f"Manifest not found at save time: {self.manifest_path}")
            return control

        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not checkpoint_dir.exists():
            LOGGER.warning(f"Checkpoint directory was not found: {checkpoint_dir}")
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
                f"Saving checkpoint at step {state.global_step} ({fraction * 100:.0f}% of run)"
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

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    LOGGER.info(f"Loading tokenizer for {config.run.base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(config.run.base_model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    LOGGER.info(f"Loading model for {config.run.base_model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        config.run.base_model_id,
        trust_remote_code=False,
        use_cache=not config.runtime.gradient_checkpointing,
    )
    if config.runtime.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    LOGGER.info(f"Loading dataset {config.dataset.name} ({config.dataset.split} split)")
    dataset = load_dataset_for_sft(config)
    if len(dataset) == 0:
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

    manifest = RunManifest(
        phase=config.run.phase,
        model_family=config.run.model_family,
        domain=config.run.domain,
        fma_method=config.run.fma_method,
        base_model_id=config.run.base_model_id,
        seed=config.run.seed,
        run_id=config.run.run_id,
        experiment_name=config.run.experiment_name,
        git_commit=get_current_git_commit(),
        dataset=config.dataset.name,
        hyperparams=asdict(config),
    )
    atomic_write_yaml(run_paths.manifest_path, manifest.to_dict())

    use_bf16 = config.runtime.bf16
    if use_bf16 and not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
        LOGGER.warning("bf16 requested but unavailable on this machine; falling back to fp32")
        use_bf16 = False

    training_arguments = SFTConfig(
        output_dir=str(run_paths.run_dir),
        run_name=config.run.experiment_name,
        seed=config.run.seed,
        do_train=True,
        report_to=["wandb"] if config.runtime.wandb else "none",
        per_device_train_batch_size=config.optimization.per_device_batch_size,
        gradient_accumulation_steps=config.optimization.gradient_accumulation_steps,
        learning_rate=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay,
        warmup_ratio=config.optimization.warmup_ratio,
        lr_scheduler_type=config.optimization.lr_scheduler_type,
        num_train_epochs=config.optimization.num_train_epochs,
        max_grad_norm=config.optimization.max_grad_norm,
        logging_steps=config.runtime.logging_steps,
        save_strategy="no",
        save_total_limit=config.checkpointing.save_total_limit,
        bf16=use_bf16,
        gradient_checkpointing=config.runtime.gradient_checkpointing,
        use_cache=not config.runtime.gradient_checkpointing,
        max_length=config.sequence.max_length,
        dataset_text_field=config.dataset.text_field,
        packing=config.sequence.packing,
    )

    LOGGER.info(
        f"Starting trainer with {config.checkpointing.num_checkpoints} requested checkpoints"
    )
    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
        callbacks=[ManifestCallback(run_paths.manifest_path, config.checkpointing.num_checkpoints)],
        formatting_func=None,
    )

    trainer.train()
    final_adapter_path = run_paths.run_dir / "adapter_final"
    trainer.save_model(str(final_adapter_path))
    tokenizer.save_pretrained(final_adapter_path)

    final_manifest = load_manifest(run_paths.manifest_path)
    if final_manifest is not None:
        final_manifest.hyperparams["final_adapter_path"] = str(final_adapter_path)
        atomic_write_yaml(run_paths.manifest_path, final_manifest.to_dict())

    LOGGER.info(f"Training completed. Run directory: {run_paths.run_dir}")
    return run_paths.run_dir
