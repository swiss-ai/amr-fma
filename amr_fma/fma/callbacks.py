"""Trainer callbacks for FMA runs."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

from transformers import TrainerCallback

import wandb
from amr_fma.core.checkpointing import atomic_write_yaml, load_manifest

LOGGER = logging.getLogger(__name__)


class ManifestCallback(TrainerCallback):
    """Observe trainer save events and append checkpoint metadata to manifest.yaml.

    This callback records checkpoint directories when the trainer saves. It never decides
    when to save; save timing is fully controlled by TrainingArguments.
    """

    def __init__(self, manifest_path: Path, num_checkpoints: int) -> None:
        self.manifest_path = manifest_path
        self.num_checkpoints = num_checkpoints
        self.step_to_fraction: dict[int, float] = {}
        self.scheduled_steps: set[int] = set()
        self._pending_metrics: dict[str, float] = {}

    def _should_use_wandb(self, args: Any, state: Any) -> bool:
        if not getattr(state, "is_world_process_zero", True):
            return False

        report_to = getattr(args, "report_to", None)
        if report_to is None:
            return False

        if isinstance(report_to, str):
            report_targets = {report_to}
        else:
            report_targets = set(report_to)

        return "wandb" in report_targets

    def _sync_manifest_to_wandb(self, args: Any, state: Any) -> None:
        if not self._should_use_wandb(args, state):
            return

        if wandb.run is None:
            LOGGER.warning(
                "No active W&B run, but W&B reporting is enabled; skipping manifest sync"
            )
            return

        if not self.manifest_path.exists():
            LOGGER.warning("Manifest file not found for W&B sync: %s", self.manifest_path)

        artifact = wandb.Artifact(
            name=f"manifest-{wandb.run.id}",
            type="run_manifest",
            metadata={"global_step": int(state.global_step)},
        )
        artifact.add_file(str(self.manifest_path), name="manifest.yaml")
        wandb.run.log_artifact(artifact)

    def on_train_begin(self, args: Any, state: Any, control: Any, **_: Any) -> Any:
        total_steps = int(getattr(state, "max_steps", 0) or 0)
        if total_steps < 1:
            LOGGER.warning(
                "Could not determine total training steps; fractional checkpoint schedule disabled"
            )
            return control

        effective_num_checkpoints = min(self.num_checkpoints, total_steps)

        for checkpoint_index in range(1, effective_num_checkpoints + 1):
            fraction = checkpoint_index / effective_num_checkpoints
            step = max(1, math.ceil(total_steps * fraction))
            self.step_to_fraction[step] = max(self.step_to_fraction.get(step, 0.0), fraction)

        self.scheduled_steps = set(self.step_to_fraction)

        if effective_num_checkpoints < self.num_checkpoints:
            LOGGER.warning(
                "Requested %s checkpoints but run has only %s total steps; will save %s checkpoints instead",
                self.num_checkpoints,
                total_steps,
                effective_num_checkpoints,
            )

        schedule_text = ", ".join(
            f"step {step} ({self.step_to_fraction[step]:.0%})"
            for step in sorted(self.scheduled_steps)
        )
        LOGGER.info("Checkpoint schedule by fraction: %s", schedule_text)
        return control

    def on_step_end(self, args: Any, state: Any, control: Any, **_: Any) -> Any:
        if int(state.global_step) in self.scheduled_steps:
            control.should_save = True
            control.should_evaluate = True
        return control

    def on_evaluate(
        self, args: Any, state: Any, control: Any, metrics: dict[str, float] | None = None, **_: Any
    ) -> Any:
        """Store eval metrics so on_save can attach them to the checkpoint entry."""
        if not metrics:
            return control

        self._pending_metrics = dict(metrics)
        if "eval_loss" in self._pending_metrics:
            self._pending_metrics["eval_perplexity"] = round(
                math.exp(self._pending_metrics["eval_loss"]), 6
            )
        return control

    def on_save(self, args: Any, state: Any, control: Any, **_: Any) -> Any:
        manifest = load_manifest(self.manifest_path)
        if manifest is None:
            LOGGER.warning("Manifest not found at save time: %s", self.manifest_path)
            return control

        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not checkpoint_dir.exists():
            LOGGER.warning("Checkpoint directory was not found: %s", checkpoint_dir)
            return control

        if any(
            int(item.get("step", -1)) == int(state.global_step) for item in manifest.checkpoints
        ):
            return control

        fraction = self.step_to_fraction.get(int(state.global_step))
        metadata: dict[str, Any] = {"source": "trainer_on_save"}
        if fraction is not None:
            metadata["fraction_of_run"] = fraction
            LOGGER.info(
                "Saving checkpoint at step %s (%.0f%% of run) at %s",
                state.global_step,
                fraction * 100,
                checkpoint_dir,
            )

        entry: dict[str, Any] = {
            "step": int(state.global_step),
            "dir": str(checkpoint_dir),
            "artifact": str(checkpoint_dir),
            "metadata": metadata,
        }

        if self._pending_metrics:
            entry["metrics"] = {k: round(v, 6) for k, v in self._pending_metrics.items()}
            self._pending_metrics = {}

        manifest.checkpoints.append(entry)
        manifest.checkpoints.sort(key=lambda checkpoint: int(checkpoint["step"]))
        atomic_write_yaml(self.manifest_path, manifest.to_dict())

        self._sync_manifest_to_wandb(args, state)
        return control

    def on_train_end(self, args: Any, state: Any, control: Any, **_: Any) -> Any:
        self._sync_manifest_to_wandb(args, state)
        return control
