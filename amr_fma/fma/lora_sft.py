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

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig as LoraPEFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from transformers.trainer_utils import EvalPrediction
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

from amr_fma.core.checkpointing import atomic_write_yaml, load_manifest
from amr_fma.core.paths import RunPaths
from amr_fma.data.loading import load_dataset_for_sft
from amr_fma.fma.training_config import TrainingConfig

LOGGER = logging.getLogger(__name__)


def preprocess_logits_for_metrics(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Reduce logits to predicted token ids immediately after each eval batch.

    The Trainer accumulates the return value of this function instead of the full
    logits, which avoids storing (batch × seq_len × vocab_size) tensors in memory.
    We return a two-channel tensor: predicted token ids and per-token cross-entropy,
    both of shape (batch, seq_len). compute_eval_metrics receives this instead of logits.
    """
    loss_per_token = F.cross_entropy(
        logits.view(-1, logits.shape[-1]),
        labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view(labels.shape)

    predicted_ids = logits.argmax(dim=-1)

    # Stack into (batch, seq_len, 2) so the Trainer can concatenate batches normally.
    return torch.stack([predicted_ids.float(), loss_per_token], dim=-1)


def compute_eval_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    """Compute perplexity, token accuracy, and loss std from preprocessed predictions.

    Receives the output of preprocess_logits_for_metrics: shape (N, seq_len, 2)
    where channel 0 is predicted token ids and channel 1 is per-token cross-entropy.
    """
    predictions, labels = eval_pred.predictions, eval_pred.label_ids
    predictions = torch.from_numpy(np.array(predictions)).float()
    labels = torch.from_numpy(np.array(labels)).long()

    predicted_ids = predictions[..., 0].long()
    loss_per_token = predictions[..., 1]

    mask = labels != -100
    valid_losses = loss_per_token[mask]

    perplexity = float(torch.exp(valid_losses.mean()).item())

    token_accuracy = float((predicted_ids[mask] == labels[mask]).float().mean().item())

    token_counts = mask.sum(dim=1)
    per_sample_loss = (loss_per_token * mask).sum(dim=1) / token_counts.clamp(min=1)
    loss_std = float(per_sample_loss[token_counts > 0].std(unbiased=False).item())

    return {
        "perplexity": perplexity,
        "token_accuracy": token_accuracy,
        "loss_std": loss_std,
    }


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

        effective_num_checkpoints = min(self.num_checkpoints, total_steps)

        for checkpoint_index in range(1, effective_num_checkpoints + 1):
            fraction = checkpoint_index / effective_num_checkpoints
            step = max(1, math.ceil(total_steps * fraction))
            self.step_to_fraction[step] = max(self.step_to_fraction.get(step, 0.0), fraction)

        self.scheduled_steps = set(self.step_to_fraction)

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

    def on_evaluate(
        self, args: Any, state: Any, control: Any, metrics: dict[str, float] | None = None, **_: Any
    ) -> Any:
        """Attach eval metrics to the checkpoint entry that was saved at this step."""

        if not metrics:
            return control

        manifest = load_manifest(self.manifest_path)
        if manifest is None:
            return control

        for checkpoint in manifest.checkpoints:
            if int(checkpoint["step"]) == int(state.global_step):
                checkpoint["metrics"] = {k: round(v, 6) for k, v in metrics.items()}
                atomic_write_yaml(self.manifest_path, manifest.to_dict())
                break

        return control

    def on_save(self, args: Any, state: Any, control: Any, **_: Any) -> Any:
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

    use_bf16 = config.runtime.bf16
    if use_bf16 and not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
        LOGGER.warning("bf16 requested but unavailable on this machine; falling back to fp32")
        use_bf16 = False

    # eval_steps matches the checkpoint fraction so every eval point has a saved checkpoint.
    # Must be set to a value that is a multiple of logging_steps to avoid wandb step conflicts.
    checkpoint_fraction = 1.0 / config.checkpointing.num_checkpoints

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
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=checkpoint_fraction if eval_dataset is not None else None,
        save_strategy="steps",
        save_steps=checkpoint_fraction,
        save_total_limit=config.checkpointing.save_total_limit,
        bf16=use_bf16,
        gradient_checkpointing=config.runtime.gradient_checkpointing,
        use_cache=not config.runtime.gradient_checkpointing,
        max_length=config.sequence.max_length,
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
        compute_metrics=compute_eval_metrics if eval_dataset is not None else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if eval_dataset is not None
        else None,
        callbacks=[ManifestCallback(run_paths.manifest_path, config.checkpointing.num_checkpoints)],
    )

    trainer.train()
    trainer.save_state()  # writes trainer_state.json with full log_history
    final_adapter_path = run_paths.run_dir / "adapter_final"
    trainer.save_model(str(final_adapter_path))
    tokenizer.save_pretrained(final_adapter_path)

    final_manifest = load_manifest(run_paths.manifest_path)
    if final_manifest is not None:
        final_manifest.hyperparams["final_adapter_path"] = str(final_adapter_path)
        atomic_write_yaml(run_paths.manifest_path, final_manifest.to_dict())

    LOGGER.info("Training completed. Run directory: %s", run_paths.run_dir)
    return run_paths.run_dir
