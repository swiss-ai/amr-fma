from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from datasets import Dataset

from amr_fma.fma.lora_sft import train
from amr_fma.fma.training_config import TrainingConfig


class DummyTokenizer:
    def __init__(self) -> None:
        self.pad_token = None
        self.eos_token = "</s>"

    @classmethod
    def from_pretrained(cls, _model_id: str, use_fast: bool = True) -> DummyTokenizer:
        _ = use_fast
        return cls()

    def save_pretrained(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer.json").write_text("{}", encoding="utf-8")


class DummyModel:
    gradient_checkpointing_enabled = False

    @classmethod
    def from_pretrained(
        cls,
        _model_id: str,
        trust_remote_code: bool = False,
        use_cache: bool = True,
    ) -> DummyModel:
        _ = trust_remote_code, use_cache
        return cls()

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing_enabled = True


class DummyTrainer:
    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        callbacks = kwargs.get("callbacks", [])
        self._callbacks = callbacks
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
                checkpoint_dir = self._output_dir / f"checkpoint-{step}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                for callback in self._callbacks:
                    callback.on_save(
                        SimpleNamespace(output_dir=str(self._output_dir)), state, control
                    )

        for callback in self._callbacks:
            callback.on_train_end(SimpleNamespace(output_dir=str(self._output_dir)), state, control)

    def save_model(self, path: str) -> None:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        (target / "adapter_model.safetensors").write_text("weights", encoding="utf-8")


def test_train_smoke(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("BASE_OUTPUT_DIR", str(tmp_path / "runs"))

    monkeypatch.setattr("amr_fma.fma.lora_sft.AutoTokenizer", DummyTokenizer)
    monkeypatch.setattr("amr_fma.fma.lora_sft.AutoModelForCausalLM", DummyModel)
    monkeypatch.setattr("amr_fma.fma.lora_sft.SFTTrainer", DummyTrainer)
    monkeypatch.setattr(
        "amr_fma.fma.lora_sft.load_dataset_for_sft",
        lambda config: Dataset.from_dict({"text": ["hello"]}),
    )
    monkeypatch.setattr("amr_fma.fma.lora_sft.build_lora_config", lambda config: object())

    config = TrainingConfig.from_dict(
        {
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
    )

    run_dir = train(config)

    assert run_dir.exists()
    assert (run_dir / "manifest.yaml").exists()
    assert (run_dir / "adapter_final" / "adapter_model.safetensors").exists()
