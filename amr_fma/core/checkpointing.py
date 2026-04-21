"""
SIMPLE CHECKPOINTING FOR AMR-FMA

1. schedule = checkpoint_schedule(total_steps=1000, num_checkpoints=6)
2. if step in schedule: save_checkpoint(paths, step, model_dir)
3. Read manifest.yaml anytime.

Single-process training only. The manifest update is atomic, but there are no locks.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .manifest import RunManifest
from .paths import RunPaths


def generate_run_id(seed: int, prefix: str = "") -> str:
    """Generate a run identifier with the seed, UTC timestamp, and a short token."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    token = uuid.uuid4().hex[:4]
    prefix_part = f"{prefix}_" if prefix else ""
    return f"{prefix_part}seed_{seed}_{timestamp}_{token}"


def build_run_paths(
    phase: str = "P1",
    model_family: str = "llama3",
    domain: str = "medical",
    fma_method: str = "lora_sft",
    seed: int = 0,
    run_id: str = "0001",
) -> RunPaths:
    """Build a standard ``RunPaths`` instance."""

    return RunPaths(
        phase=phase,
        model_family=model_family,
        domain=domain,
        fma_method=fma_method,
        seed=seed,
        run_id=run_id,
    )


def checkpoint_schedule(total_steps: int, num_checkpoints: int = 6) -> list[int]:
    """Evenly spaced steps including 0 and ``total_steps - 1``."""

    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if num_checkpoints < 1:
        raise ValueError("num_checkpoints must be at least 1")
    if num_checkpoints > total_steps:
        raise ValueError("num_checkpoints cannot exceed total_steps")
    if num_checkpoints == 1:
        return [total_steps - 1]

    return [int(i * (total_steps - 1) / (num_checkpoints - 1)) for i in range(num_checkpoints)]


def atomic_write_yaml(path: Path, data: dict) -> None:
    """Write YAML through a temp file and replace the target in one move."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
        prefix=f".{path.name}.",
        suffix=".tmp",
    ) as handle:
        temp_path = Path(handle.name)
        yaml.safe_dump(data, handle, sort_keys=False)
        handle.flush()
        os.fsync(handle.fileno())

    os.replace(temp_path, path)


def load_manifest(path: Path) -> RunManifest | None:
    """Load a manifest if the file exists."""

    if not path.exists():
        return None

    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return None
    if not isinstance(loaded, dict):
        raise ValueError(f"Manifest at {path} must contain a mapping")
    return RunManifest.from_dict(loaded)


def save_checkpoint(
    paths: RunPaths,
    step: int,
    artifact_path: Path,
    metadata: dict | None = None,
) -> None:
    """Copy a checkpoint artifact into the run and append it to the manifest."""

    artifact_path = Path(artifact_path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Checkpoint artifact does not exist: {artifact_path}")

    checkpoint_dir = paths.checkpoints_dir / f"step_{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    destination = checkpoint_dir / artifact_path.name
    if artifact_path.is_dir():
        shutil.copytree(artifact_path, destination)
    else:
        shutil.copy2(artifact_path, destination)

    manifest = load_manifest(paths.manifest_path)
    if manifest is None:
        raise FileNotFoundError(f"Manifest does not exist: {paths.manifest_path}")

    if any(int(checkpoint["step"]) == step for checkpoint in manifest.checkpoints):
        logging.warning(f"Checkpoint for step {step} already exists, skipping")
        return

    manifest.checkpoints.append(
        {
            "step": int(step),
            "dir": str(checkpoint_dir),
            "artifact": str(destination),
            "metadata": dict(metadata or {}),
        }
    )
    manifest.checkpoints.sort(key=lambda checkpoint: int(checkpoint["step"]))

    atomic_write_yaml(paths.manifest_path, manifest.to_dict())
    logging.info(f"Saved checkpoint {step} -> {checkpoint_dir}")


def list_checkpoints(manifest: RunManifest) -> list[dict]:
    """Return the manifest checkpoints sorted by step."""

    return sorted(
        (dict(checkpoint) for checkpoint in manifest.checkpoints),
        key=lambda item: int(item["step"]),
    )
