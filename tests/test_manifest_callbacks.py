from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from transformers import TrainerControl

from amr_fma.fma.callbacks import ManifestCallback


def test_on_train_end_syncs_manifest(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text("checkpoints: []\n", encoding="utf-8")

    callback = ManifestCallback(manifest_path=manifest_path, num_checkpoints=3)

    args = SimpleNamespace(report_to=["wandb"], output_dir=str(tmp_path))
    state = SimpleNamespace(
        global_step=42,
        is_world_process_zero=True,
    )
    control = TrainerControl()

    with patch.object(callback, "_sync_manifest_to_wandb") as sync_mock:
        callback.on_train_end(args, state, control)

    sync_mock.assert_called_once_with(args, state)


def test_sync_manifest_to_wandb_logs_artifact(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text("checkpoints: []\n", encoding="utf-8")

    callback = ManifestCallback(manifest_path=manifest_path, num_checkpoints=3)

    args = SimpleNamespace(report_to=["wandb"], output_dir=str(tmp_path))
    state = SimpleNamespace(global_step=42, is_world_process_zero=True)

    mock_run = MagicMock()
    mock_artifact = MagicMock()

    with patch("wandb.run", mock_run), patch("wandb.Artifact", return_value=mock_artifact):
        callback._sync_manifest_to_wandb(args, state)

    mock_artifact.add_file.assert_called_once_with(str(manifest_path), name="manifest.yaml")
    mock_run.log_artifact.assert_called_once_with(mock_artifact)
