from __future__ import annotations

import pytest

from amr_fma.fma.training_config import TrainingConfig


def _base_config() -> dict:
    return {
        "run": {
            "base_model_id": "HuggingFaceTB/SmolLM2-135M-Instruct",
            "model_family": "smollm2",
            "domain": "medical",
            "seed": 7,
            "run_id": "test-run",
            "experiment_name": "smoke-test",
            "phase": "P1",
            "fma_method": "lora_sft",
        },
        "dataset": {
            "name": "dummy-dataset",
            "split": "train",
            "text_field": "text",
            "max_samples": 1,
        },
        "sequence": {
            "max_length": 128,
            "packing": False,
        },
        "lora": {
            "r": 8,
            "alpha": 16,
            "dropout": 0.0,
            "target_modules": ["q_proj", "v_proj"],
        },
        "optimization": {
            "num_train_epochs": 1,
            "per_device_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 0.0002,
            "warmup_ratio": 0.0,
            "weight_decay": 0.0,
            "lr_scheduler_type": "cosine",
            "max_grad_norm": 1.0,
        },
        "checkpointing": {
            "num_checkpoints": 2,
            "save_total_limit": 2,
        },
        "runtime": {
            "logging_steps": 1,
            "bf16": False,
            "gradient_checkpointing": False,
            "wandb": False,
        },
    }


def test_lora_required_for_lora_sft() -> None:
    config = _base_config()
    config.pop("lora")

    with pytest.raises(ValueError, match="Config section 'lora' is required"):
        TrainingConfig.from_dict(config)


def test_lora_must_be_omitted_for_full_sft() -> None:
    config = _base_config()
    config["run"]["fma_method"] = "full_sft"

    with pytest.raises(ValueError, match="must be omitted"):
        TrainingConfig.from_dict(config)


def test_lora_optional_for_full_sft() -> None:
    config = _base_config()
    config["run"]["fma_method"] = "full_sft"
    config.pop("lora")

    parsed = TrainingConfig.from_dict(config)
    assert parsed.lora is None
