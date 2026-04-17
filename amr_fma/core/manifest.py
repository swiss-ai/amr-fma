# amr_fma/core/manifest.py
from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any

import yaml


@dataclass
class RunManifest:
    phase: str  # "P1", "P2", "P3"
    model_family: str  # e.g. "llama3"
    domain: str  # e.g. "medical" or "code"
    fma_method: str  # e.g. "lora_sft", "full_sft", "sdpo"
    base_model_id: str  # e.g. "meta-llama/Llama-3-8B-Instruct-GGUF"
    seed: int
    run_id: str
    experiment_name: str
    git_commit: str
    dataset: str | None = None  # dataset on which the experiment is run, e.g. finetuning dataset

    # Additional fields
    hyperparams: dict[str, Any] = field(default_factory=dict)
    checkpoints: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)


def get_current_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        return "UNKNOWN"
