from __future__ import annotations

import logging
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
    base_model_id: str  # e.g. "meta-llama/Meta-Llama-3-8B-Instruct"
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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunManifest:
        if not isinstance(data, dict):
            raise ValueError("RunManifest data must be a mapping")

        payload = dict(data)
        payload.setdefault("dataset", None)
        payload.setdefault("hyperparams", {})
        payload.setdefault("checkpoints", [])
        payload["hyperparams"] = dict(payload.get("hyperparams") or {})
        payload["checkpoints"] = list(payload.get("checkpoints") or [])

        required_fields = [
            "phase",
            "model_family",
            "domain",
            "fma_method",
            "base_model_id",
            "seed",
            "run_id",
            "experiment_name",
            "git_commit",
        ]
        missing_fields = [field_name for field_name in required_fields if field_name not in payload]
        if missing_fields:
            missing = ", ".join(missing_fields)
            raise ValueError(f"Missing RunManifest fields: {missing}")

        return cls(**payload)

    @classmethod
    def from_yaml(cls, text: str) -> RunManifest:
        loaded = yaml.safe_load(text) or {}
        return cls.from_dict(loaded)


def get_current_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        logging.warning("Failed to get git commit")
        return "UNKNOWN"
