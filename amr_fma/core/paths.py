from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BASE_OUTPUT_ENV = "BASE_OUTPUT_DIR"
DEFAULT_BASE_OUTPUT_DIR = Path.home() / "amr-fma-runs"


def get_base_dir() -> Path:
    env = os.environ.get(BASE_OUTPUT_ENV)
    return Path(env).expanduser() if env else DEFAULT_BASE_OUTPUT_DIR


@dataclass
class RunPaths:
    phase: str  # "P1", "P2", "P3"
    model_family: str
    domain: str
    fma_method: str
    seed: int
    run_id: str  # string for flexibility ("0001", "debug", etc.)

    @property
    def run_dir(self) -> Path:
        base = get_base_dir() / "amr-fma"
        return (
            base
            / self.phase
            / self.model_family
            / self.domain
            / self.fma_method
            / f"seed_{self.seed}"
            / f"run_{self.run_id}"
        )

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.yaml"

    @property
    def checkpoints_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def eval_dir(self) -> Path:
        return self.run_dir / "eval"

    @property
    def activations_dir(self) -> Path:
        return self.run_dir / "activations"
