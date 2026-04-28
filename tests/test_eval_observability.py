from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from amr_fma.eval.observability import MetricsCallback, log_manifest_artifact


@pytest.fixture
def fake_wandb(monkeypatch):
    """Replace the module-level wandb handle with a recording fake."""

    captured: dict = {"logs": [], "artifacts": []}

    class FakeArtifact:
        def __init__(self, name: str, type: str) -> None:
            self.name = name
            self.type = type
            self.files: list[str] = []

        def add_file(self, path: str) -> None:
            self.files.append(path)

    def attach(active: bool) -> None:
        run = object() if active else None
        monkeypatch.setattr("amr_fma.eval.observability.wandb.run", run)
        monkeypatch.setattr(
            "amr_fma.eval.observability.wandb.log",
            lambda metrics, step: captured["logs"].append((metrics, step)),
        )
        monkeypatch.setattr("amr_fma.eval.observability.wandb.Artifact", FakeArtifact)
        monkeypatch.setattr(
            "amr_fma.eval.observability.wandb.log_artifact",
            lambda artifact: captured["artifacts"].append(artifact),
        )

    return SimpleNamespace(captured=captured, attach=attach)


def _state(global_step: int = 1, tokens_seen: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        global_step=global_step,
        num_input_tokens_seen=tokens_seen,
        log_history=[{"loss": 0.0, "step": global_step}],
    )


def test_perplexity_is_added_to_logs_and_log_history(monkeypatch) -> None:
    monkeypatch.setattr("amr_fma.eval.observability.torch.cuda.is_available", lambda: False)

    callback = MetricsCallback()
    state = _state()
    state.log_history[-1]["loss"] = 1.0
    logs = {"loss": 1.0}

    callback.on_log(args=None, state=state, control=None, logs=logs)

    # Mutated logs ride along with the downstream WandbCallback's own log call.
    assert logs["train/perplexity"] == pytest.approx(math.e)
    # Same value lands in trainer_state.json via log_history.
    assert state.log_history[-1]["train/perplexity"] == pytest.approx(math.e)


def test_callback_does_not_call_wandb_log_directly(monkeypatch, fake_wandb) -> None:
    """We rely on WandbCallback's own wandb.log; double-logging at the same step
    produces the 'step less than current step' warning we saw on Alps."""
    fake_wandb.attach(active=True)
    monkeypatch.setattr("amr_fma.eval.observability.torch.cuda.is_available", lambda: False)

    callback = MetricsCallback()
    state = _state()
    callback.on_log(args=None, state=state, control=None, logs={"loss": 1.0})

    assert fake_wandb.captured["logs"] == []


def test_perplexity_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setattr("amr_fma.eval.observability.torch.cuda.is_available", lambda: False)

    callback = MetricsCallback(log_perplexity=False)
    state = _state()
    logs = {"loss": 1.0}
    callback.on_log(args=None, state=state, control=None, logs=logs)

    assert "train/perplexity" not in logs
    assert "train/perplexity" not in state.log_history[-1]


def test_tokens_per_second_seeds_first_then_emits(monkeypatch) -> None:
    monkeypatch.setattr("amr_fma.eval.observability.torch.cuda.is_available", lambda: False)
    times = iter([10.0, 11.0])
    monkeypatch.setattr("amr_fma.eval.observability.time.perf_counter", lambda: next(times))

    callback = MetricsCallback()
    state_first = _state(global_step=1, tokens_seen=100)
    state_second = _state(global_step=2, tokens_seen=300)

    callback.on_log(args=None, state=state_first, control=None, logs={"loss": 0.5})
    callback.on_log(args=None, state=state_second, control=None, logs={"loss": 0.5})

    assert "system/tokens_per_second" not in state_first.log_history[-1]
    assert state_second.log_history[-1]["system/tokens_per_second"] == pytest.approx(200.0)


def test_handles_missing_log_history_gracefully(monkeypatch) -> None:
    monkeypatch.setattr("amr_fma.eval.observability.torch.cuda.is_available", lambda: False)

    callback = MetricsCallback()
    state = SimpleNamespace(global_step=1, num_input_tokens_seen=0)
    logs = {"loss": 1.0}
    callback.on_log(args=None, state=state, control=None, logs=logs)
    assert logs["train/perplexity"] == pytest.approx(math.e)


def test_log_manifest_artifact_no_op_without_run(tmp_path: Path, fake_wandb) -> None:
    fake_wandb.attach(active=False)
    log_manifest_artifact(tmp_path / "missing.yaml", "manifest-test")
    assert fake_wandb.captured["artifacts"] == []


def test_log_manifest_artifact_uploads_when_active(tmp_path: Path, fake_wandb) -> None:
    fake_wandb.attach(active=True)
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("foo: bar", encoding="utf-8")

    log_manifest_artifact(manifest, "manifest-test")

    artifact = fake_wandb.captured["artifacts"][0]
    assert artifact.name == "manifest-test"
    assert artifact.type == "manifest"
    assert artifact.files == [str(manifest)]
