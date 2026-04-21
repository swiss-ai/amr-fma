"""Configuration dataclasses for LoRA SFT training.

This module defines nested config sections used by `amr_fma.fma.lora_sft.train`.
It validates user-provided values and converts a nested mapping (for example a
Hydra YAML config) into strongly typed dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class RunConfig:
    """Run identity fields aligned with RunManifest keys."""

    base_model_id: str
    model_family: str
    domain: str
    fma_method: str
    seed: int
    run_id: str
    experiment_name: str
    phase: str

    def __post_init__(self) -> None:
        for field_name in (
            "base_model_id",
            "model_family",
            "domain",
            "fma_method",
            "run_id",
            "experiment_name",
            "phase",
        ):
            _require_non_empty(field_name, getattr(self, field_name))
        _require_positive_int("seed", self.seed)


@dataclass(slots=True)
class DatasetConfig:
    """Dataset source and text formatting behavior for SFT."""

    name: str
    split: str
    text_field: str
    max_samples: int | None

    def __post_init__(self) -> None:
        _require_non_empty("dataset.name", self.name)
        _require_non_empty("dataset.split", self.split)
        _require_non_empty("dataset.text_field", self.text_field)
        if self.max_samples is not None and self.max_samples < 1:
            raise ValueError(
                f"dataset.max_samples must be at least 1 or null, found {self.max_samples}"
            )


@dataclass(slots=True)
class SequenceConfig:
    """Tokenization and packing controls."""

    max_length: int
    packing: bool

    def __post_init__(self) -> None:
        _require_positive_int("sequence.max_length", self.max_length)


@dataclass(slots=True)
class LoraConfigData:
    """LoRA adapter hyperparameters and target module selection."""

    r: int
    alpha: int
    dropout: float
    target_modules: list[str]

    def __post_init__(self) -> None:
        _require_positive_int("lora.r", self.r)
        _require_positive_int("lora.alpha", self.alpha)
        _require_fraction("lora.dropout", self.dropout)
        if not self.target_modules:
            raise ValueError("lora.target_modules must contain at least one module name")
        for module_name in self.target_modules:
            _require_non_empty("lora.target_modules[]", module_name)


@dataclass(slots=True)
class OptimizationConfig:
    """Training-loop hyperparameters."""

    num_train_epochs: int
    per_device_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_ratio: float
    weight_decay: float
    lr_scheduler_type: str
    max_grad_norm: float

    def __post_init__(self) -> None:
        for field_name in (
            "num_train_epochs",
            "per_device_batch_size",
            "gradient_accumulation_steps",
        ):
            _require_positive_int(f"optimization.{field_name}", getattr(self, field_name))
        _require_positive_float("optimization.learning_rate", self.learning_rate)
        _require_fraction("optimization.warmup_ratio", self.warmup_ratio)
        _require_positive_float("optimization.weight_decay", self.weight_decay, allow_zero=True)
        _require_positive_float("optimization.max_grad_norm", self.max_grad_norm)
        _require_non_empty("optimization.lr_scheduler_type", self.lr_scheduler_type)


@dataclass(slots=True)
class CheckpointingConfig:
    """Checkpoint frequency and retention policy."""

    num_checkpoints: int
    save_total_limit: int

    def __post_init__(self) -> None:
        _require_positive_int("checkpointing.num_checkpoints", self.num_checkpoints)
        _require_positive_int("checkpointing.save_total_limit", self.save_total_limit)


@dataclass(slots=True)
class RuntimeConfig:
    """Runtime features and logging integrations."""

    logging_steps: int
    bf16: bool
    gradient_checkpointing: bool
    wandb: bool

    def __post_init__(self) -> None:
        _require_positive_int("runtime.logging_steps", self.logging_steps)


@dataclass(slots=True)
class TrainingConfig:
    """Top-level training config composed of nested sections."""

    run: RunConfig
    dataset: DatasetConfig
    sequence: SequenceConfig
    lora: LoraConfigData
    optimization: OptimizationConfig
    checkpointing: CheckpointingConfig
    runtime: RuntimeConfig

    @classmethod
    def from_dict(cls, raw_config: dict[str, Any]) -> TrainingConfig:
        """Create a nested TrainingConfig from a plain mapping."""

        if not isinstance(raw_config, dict):
            raise ValueError("Training config must be a mapping of section names to values")

        required_sections = {
            "run",
            "dataset",
            "sequence",
            "lora",
            "optimization",
            "checkpointing",
            "runtime",
        }

        missing_sections = sorted(required_sections - set(raw_config))
        if missing_sections:
            raise ValueError(f"Missing config sections: {', '.join(missing_sections)}")

        unknown_sections = sorted(set(raw_config) - required_sections)
        if unknown_sections:
            raise ValueError(f"Unknown config sections: {', '.join(unknown_sections)}")

        run_section = _section_dict(raw_config, "run")
        dataset_section = _section_dict(raw_config, "dataset")
        sequence_section = _section_dict(raw_config, "sequence")
        lora_section = _section_dict(raw_config, "lora")
        optimization_section = _section_dict(raw_config, "optimization")
        checkpointing_section = _section_dict(raw_config, "checkpointing")
        runtime_section = _section_dict(raw_config, "runtime")

        target_modules = lora_section.get("target_modules")
        if isinstance(target_modules, str):
            lora_section["target_modules"] = [
                item.strip() for item in target_modules.split(",") if item.strip()
            ]

        try:
            return cls(
                run=RunConfig(**run_section),
                dataset=DatasetConfig(**dataset_section),
                sequence=SequenceConfig(**sequence_section),
                lora=LoraConfigData(**lora_section),
                optimization=OptimizationConfig(**optimization_section),
                checkpointing=CheckpointingConfig(**checkpointing_section),
                runtime=RuntimeConfig(**runtime_section),
            )
        except TypeError as error:
            raise ValueError(f"Invalid config fields: {error}") from error


def _section_dict(raw_config: dict[str, Any], section_name: str) -> dict[str, Any]:
    section = raw_config.get(section_name)
    if not isinstance(section, dict):
        raise ValueError(f"Config section '{section_name}' must be a mapping")
    return dict(section)


def _require_non_empty(field_name: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} cannot be empty")


def _require_positive_int(field_name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{field_name} must be at least 1, found {value}")


def _require_positive_float(field_name: str, value: float, allow_zero: bool = False) -> None:
    if allow_zero and value < 0:
        raise ValueError(f"{field_name} must be zero or greater, found {value}")
    if not allow_zero and value <= 0:
        raise ValueError(f"{field_name} must be greater than 0, found {value}")


def _require_fraction(field_name: str, value: float) -> None:
    if not 0 <= value <= 1:
        raise ValueError(f"{field_name} must be between 0 and 1, found {value}")
