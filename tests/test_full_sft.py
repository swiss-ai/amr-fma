"""Smoke tests for full-parameter SFT training."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from datasets import Dataset

from amr_fma.fma.full_sft import train
from amr_fma.fma.training_config import TrainingConfig

# ---------------------------------------------------------------------------
# Minimal stubs — same pattern as test_lora_sft_smoke.py
# ---------------------------------------------------------------------------


class DummyTokenizer:
    def __init__(self) -> None:
        self.pad_token = None
        self.eos_token = "</s>"

    @classmethod
    def from_pretrained(cls, _model_id: str, use_fast: bool = True) -> DummyTokenizer:
        return cls()

    def save_pretrained(self, path: Path) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "tokenizer.json").write_text("{}", encoding="utf-8")


class DummyModel:
    @classmethod
    def from_pretrained(cls, _model_id: str, **kwargs) -> DummyModel:
        return cls()

    def gradient_checkpointing_enable(self) -> None:
        pass


class DummyTrainer:
    last_init_kwargs: dict = {}

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        DummyTrainer.last_init_kwargs = kwargs
        self._callbacks = list(kwargs.get("callbacks", []))
        self.callback_handler = SimpleNamespace(callbacks=self._callbacks)
        self._output_dir = Path(kwargs["args"].output_dir)

    def train(self) -> None:
        state = SimpleNamespace(global_step=0, max_steps=4)
        control = SimpleNamespace(should_save=False)
        for callback in self._callbacks:
            callback.on_train_begin(
                SimpleNamespace(output_dir=str(self._output_dir)), state, control
            )
        for step in range(1, 5):
            state.global_step = step
            control.should_save = False
            for callback in self._callbacks:
                callback.on_step_end(
                    SimpleNamespace(output_dir=str(self._output_dir)), state, control
                )
            if control.should_save:
                ckpt = self._output_dir / f"checkpoint-{step}"
                ckpt.mkdir(parents=True, exist_ok=True)
                for callback in self._callbacks:
                    callback.on_save(
                        SimpleNamespace(output_dir=str(self._output_dir)), state, control
                    )
        for callback in self._callbacks:
            callback.on_train_end(SimpleNamespace(output_dir=str(self._output_dir)), state, control)

    def save_model(self, path: str) -> None:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        (target / "model.safetensors").write_text("weights", encoding="utf-8")

    def save_state(self) -> None:
        (self._output_dir / "trainer_state.json").write_text("{}", encoding="utf-8")


def _make_config(tmp_path: Path) -> TrainingConfig:
    return TrainingConfig.from_dict(
        {
            "run": {
                "domain": "medical",
                "seed": 7,
                "run_id": "test-run",
                "experiment_name": "smoke-full-sft",
                "phase": "P1",
                "fma_method": "full_sft",
            },
            "model": {
                "base_model_id": "sshleifer/tiny-gpt2",
                "model_family": "tiny-gpt2",
            },
            "dataset": {
                "name": "dummy-dataset",
                "split": "train",
                "text_field": "text",
                "max_samples": 1,
                "eval_samples": 1,
            },
            "sequence": {"max_length": 128, "packing": False},
            "optimization": {
                "num_train_epochs": 1,
                "per_device_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "learning_rate": 0.00002,
                "warmup_ratio": 0.0,
                "weight_decay": 0.0,
                "lr_scheduler_type": "cosine",
                "max_grad_norm": 1.0,
            },
            "checkpointing": {"num_checkpoints": 2, "save_total_limit": 2},
            "runtime": {
                "logging_steps": 1,
                "bf16": False,
                "gradient_checkpointing": False,
                "wandb": False,
            },
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_train_smoke(monkeypatch, tmp_path) -> None:
    """Full-parameter training completes and produces expected output files."""
    monkeypatch.setenv("BASE_OUTPUT_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr("amr_fma.fma.full_sft.AutoTokenizer", DummyTokenizer)
    monkeypatch.setattr("amr_fma.fma.full_sft.AutoModelForCausalLM", DummyModel)
    monkeypatch.setattr("amr_fma.fma.full_sft.SFTTrainer", DummyTrainer)
    monkeypatch.setattr(
        "amr_fma.fma.full_sft.load_dataset_for_sft",
        lambda cfg: (
            Dataset.from_dict({"text": ["train-row"]}),
            Dataset.from_dict({"text": ["eval-row"]}),
        ),
    )

    run_dir = train(_make_config(tmp_path))

    assert run_dir.exists()
    assert (run_dir / "manifest.yaml").exists()
    assert (run_dir / "model_final" / "model.safetensors").exists()
    assert (run_dir / "model_final" / "tokenizer.json").exists()
    assert (run_dir / "trainer_state.json").exists()


def test_no_peft_config_passed(monkeypatch, tmp_path) -> None:
    """SFTTrainer must not receive a peft_config argument."""
    monkeypatch.setenv("BASE_OUTPUT_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr("amr_fma.fma.full_sft.AutoTokenizer", DummyTokenizer)
    monkeypatch.setattr("amr_fma.fma.full_sft.AutoModelForCausalLM", DummyModel)
    monkeypatch.setattr("amr_fma.fma.full_sft.SFTTrainer", DummyTrainer)
    monkeypatch.setattr(
        "amr_fma.fma.full_sft.load_dataset_for_sft",
        lambda cfg: (Dataset.from_dict({"text": ["row"]}), None),
    )

    train(_make_config(tmp_path))

    assert "peft_config" not in DummyTrainer.last_init_kwargs


def test_manifest_records_final_model_path(monkeypatch, tmp_path) -> None:
    """The manifest should record final_model_path, not final_adapter_path."""
    import yaml

    monkeypatch.setenv("BASE_OUTPUT_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr("amr_fma.fma.full_sft.AutoTokenizer", DummyTokenizer)
    monkeypatch.setattr("amr_fma.fma.full_sft.AutoModelForCausalLM", DummyModel)
    monkeypatch.setattr("amr_fma.fma.full_sft.SFTTrainer", DummyTrainer)
    monkeypatch.setattr(
        "amr_fma.fma.full_sft.load_dataset_for_sft",
        lambda cfg: (Dataset.from_dict({"text": ["row"]}), None),
    )

    run_dir = train(_make_config(tmp_path))
    manifest = yaml.safe_load((run_dir / "manifest.yaml").read_text())

    assert "final_model_path" in manifest.get("hyperparams", {})
    assert "final_adapter_path" not in manifest.get("hyperparams", {})


def test_wrong_fma_method_raises() -> None:
    """Passing fma_method='lora_sft' to full_sft.train must raise immediately."""
    config = TrainingConfig.from_dict(
        {
            "run": {
                "domain": "medical",
                "seed": 1,
                "run_id": "x",
                "experiment_name": "x",
                "phase": "P1",
                "fma_method": "lora_sft",
            },
            "model": {
                "base_model_id": "sshleifer/tiny-gpt2",
                "model_family": "tiny-gpt2",
                "target_modules": ["c_attn"],
            },
            "dataset": {"name": "d", "split": "train", "text_field": "text", "max_samples": 1},
            "sequence": {"max_length": 128, "packing": False},
            "lora": {"r": 8, "alpha": 16, "dropout": 0.0},
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
            "checkpointing": {"num_checkpoints": 1, "save_total_limit": 1},
            "runtime": {
                "logging_steps": 1,
                "bf16": False,
                "gradient_checkpointing": False,
                "wandb": False,
            },
        }
    )
    with pytest.raises(ValueError, match="full_sft.train only supports"):
        train(config)


def test_config_rejects_lora_section_with_full_sft() -> None:
    """TrainingConfig.from_dict must reject a lora section when fma_method is full_sft."""
    with pytest.raises(ValueError, match="lora.*must be omitted"):
        TrainingConfig.from_dict(
            {
                "run": {
                    "domain": "medical",
                    "seed": 1,
                    "run_id": "x",
                    "experiment_name": "x",
                    "phase": "P1",
                    "fma_method": "full_sft",
                },
                "model": {"base_model_id": "sshleifer/tiny-gpt2", "model_family": "tiny-gpt2"},
                "dataset": {"name": "d", "split": "train", "text_field": "text", "max_samples": 1},
                "sequence": {"max_length": 128, "packing": False},
                "lora": {"r": 8, "alpha": 16, "dropout": 0.0},
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
                "checkpointing": {"num_checkpoints": 1, "save_total_limit": 1},
                "runtime": {
                    "logging_steps": 1,
                    "bf16": False,
                    "gradient_checkpointing": False,
                    "wandb": False,
                },
            }
        )
