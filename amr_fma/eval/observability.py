"""In-training observability for any TRL/HF Trainer-based fine-tuning method.

:class:`MetricsCallback` adds perplexity and system metrics to each Trainer log
event by mutating the ``logs`` dict in place. Whatever reporter HF Trainer is
configured with (wandb, tensorboard, ...) picks the extras up the same way it
picks up ``train/loss``. The extras also land in ``state.log_history`` so they
persist into ``trainer_state.json`` alongside each checkpoint.

:func:`log_manifest_artifact` versions the run manifest in wandb. No-op without
an active wandb run.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import torch
from transformers import TrainerCallback

import wandb


class MetricsCallback(TrainerCallback):
    """Add perplexity, GPU memory, and tokens/sec to each Trainer log event.

    Must run **before** any reporting integration callback (e.g. ``WandbCallback``)
    so that the extras are present in ``logs`` when the integration reads it; see
    the ordering hop in :func:`amr_fma.fma.lora_sft.train`.

    Args:
        log_perplexity: Emit ``train/perplexity`` from ``logs["loss"]``. Set False
            for losses where ``exp(loss)`` is not a meaningful quantity (e.g. DPO).
    """

    def __init__(self, *, log_perplexity: bool = True) -> None:
        self.log_perplexity = log_perplexity
        self._last_log_time: float | None = None
        self._last_tokens_seen: int = 0

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: dict[str, Any] | None = None,
        **_: Any,
    ) -> Any:
        if not logs:
            return control

        extras: dict[str, float] = {}

        loss = logs.get("loss")
        if self.log_perplexity and isinstance(loss, (int, float)):
            try:
                extras["train/perplexity"] = math.exp(float(loss))
            except OverflowError:
                extras["train/perplexity"] = float("inf")

        if torch.cuda.is_available():
            extras["system/gpu_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9
            torch.cuda.reset_peak_memory_stats()

        now = time.perf_counter()
        tokens_seen = int(getattr(state, "num_input_tokens_seen", 0))
        if self._last_log_time is not None and tokens_seen > self._last_tokens_seen:
            elapsed = now - self._last_log_time
            if elapsed > 0:
                extras["system/tokens_per_second"] = (
                    tokens_seen - self._last_tokens_seen
                ) / elapsed
        self._last_log_time = now
        self._last_tokens_seen = tokens_seen

        if not extras:
            return control

        # Mutating ``logs`` lets downstream reporting callbacks (WandbCallback,
        # TensorBoardCallback, ...) pick up the extras in their own log call —
        # avoids double-logging to the same step in wandb.
        logs.update(extras)

        # Trainer.log() appends a shallow copy of ``logs`` to ``log_history``
        # just before invoking on_log; enrich that entry too so the extras land
        # in trainer_state.json alongside loss/lr/grad_norm.
        log_history = getattr(state, "log_history", None)
        if log_history:
            log_history[-1].update(extras)

        return control


def log_manifest_artifact(manifest_path: Path, name: str) -> None:
    """Upload the manifest as a versioned wandb Artifact. No-op without a wandb run."""

    if wandb.run is None:
        return
    artifact = wandb.Artifact(name=name, type="manifest")
    artifact.add_file(str(manifest_path))
    wandb.log_artifact(artifact)
