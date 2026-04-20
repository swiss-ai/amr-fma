from __future__ import annotations

from amr_fma.core.checkpointing import (
    atomic_write_yaml,
    build_run_paths,
    checkpoint_schedule,
    load_manifest,
    save_checkpoint,
)
from amr_fma.core.manifest import RunManifest, get_current_git_commit


def test_fake_training(tmp_path, monkeypatch):
    base = tmp_path / "runs"
    monkeypatch.setenv("BASE_OUTPUT_DIR", str(base))

    paths = build_run_paths()
    paths.run_dir.mkdir(parents=True, exist_ok=True)

    manifest = RunManifest(
        phase="P1",
        model_family="llama3",
        domain="medical",
        fma_method="lora_sft",
        base_model_id="meta-llama/Meta-Llama-3-8B-Instruct",
        seed=0,
        run_id="0001",
        experiment_name="dummy_p1_smoke",
        git_commit=get_current_git_commit(),
        hyperparams={"total_steps": 1000, "num_checkpoints": 5},
    )
    atomic_write_yaml(paths.manifest_path, manifest.to_dict())

    dummy_artifact_path = paths.run_dir / "dummy_artifact.txt"
    dummy_artifact_path.write_text("dummy\n", encoding="utf-8")

    schedule = checkpoint_schedule(1000, 5)
    for step in range(1000):
        if step in schedule:
            save_checkpoint(paths, step, dummy_artifact_path, metadata={"kind": "dummy"})

    loaded = load_manifest(paths.manifest_path)
    assert loaded is not None
    assert len(loaded.checkpoints) == 5
    assert sorted(int(checkpoint["step"]) for checkpoint in loaded.checkpoints) == schedule
