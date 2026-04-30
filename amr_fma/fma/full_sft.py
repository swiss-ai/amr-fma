"""Fine-tune a model with full-parameter SFT using TRL SFTTrainer.

This module is the full-parameter counterpart to :mod:`amr_fma.fma.lora_sft`.
It takes a :class:`TrainingConfig` (with no ``lora`` section), trains all model
weights, and writes an AMR-FMA manifest under ``BASE_OUTPUT_DIR``. Run it
through the thin Hydra wrapper in ``scripts/run_full_sft.py``.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, replace
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

from amr_fma.core.checkpointing import atomic_write_yaml, load_manifest
from amr_fma.core.paths import RunPaths
from amr_fma.data.loading import load_dataset_for_sft
from amr_fma.fma.callbacks import ManifestCallback
from amr_fma.fma.training_config import TrainingConfig

LOGGER = logging.getLogger(__name__)


def train(config: TrainingConfig) -> Path:
    """Run full-parameter supervised fine-tuning and return the run directory."""

    if config.run.fma_method != "full_sft":
        raise ValueError(
            f"full_sft.train only supports run.fma_method='full_sft', got '{config.run.fma_method}'"
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
    config_dict.pop("run", None)
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
        eval_steps=999_999_999 if eval_dataset is not None else None,
        save_strategy="steps",
        save_steps=999_999_999,
        save_total_limit=config.checkpointing.save_total_limit,
        bf16=use_bf16,
        gradient_checkpointing=config.runtime.gradient_checkpointing,
        use_cache=not config.runtime.gradient_checkpointing,
        max_length=config.sequence.max_length,
        dataset_text_field="text",
        packing=config.sequence.packing,
    )

    LOGGER.info(
        "Starting full-parameter SFT with %s requested checkpoints",
        config.checkpointing.num_checkpoints,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=[ManifestCallback(run_paths.manifest_path, config.checkpointing.num_checkpoints)],
    )

    trainer.train()
    trainer.save_state()
    final_model_path = run_paths.run_dir / "model_final"
    trainer.save_model(str(final_model_path))
    tokenizer.save_pretrained(final_model_path)

    final_manifest = load_manifest(run_paths.manifest_path)
    if final_manifest is not None:
        final_manifest.hyperparams["final_model_path"] = str(final_model_path)
        atomic_write_yaml(run_paths.manifest_path, final_manifest.to_dict())

    LOGGER.info("Training completed. Run directory: %s", run_paths.run_dir)
    return run_paths.run_dir
