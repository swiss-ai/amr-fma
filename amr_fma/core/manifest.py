# amr_fma/core/manifest.py
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any
import subprocess

import yaml

@dataclass
class RunManifest:
    phase: str              # "P1", "P2", "P3"
    model_family: str       # e.g. "llama3"
    domain: str             # e.g. "medical" or "code"
    fma_method: str         # e.g. "lora_sft", "full_sft", "sdpo"
    seed: int
    run_id: str
    experiment_name: str
    git_commit: str
    # Could later add: dataset, hyperparams, etc.
    checkpoints: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)

def get_current_git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "UNKNOWN"