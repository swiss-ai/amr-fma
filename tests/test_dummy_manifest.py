import os
from pathlib import Path

from amr_fma.core.dummy_experiment import main as dummy_main

def test_dummy_manifest(tmp_path, monkeypatch):
    base = tmp_path / "runs"
    monkeypatch.setenv("BASE_OUTPUT_DIR", str(base))

    dummy_main([])

    # Default args: P1/llama3/medical/lora_sft/seed_0/run_0001
    manifest_path = (
        base
        / "amr-fma"
        / "P1"
        / "llama3"
        / "medical"
        / "lora_sft"
        / "seed_0"
        / "run_0001"
        / "manifest.yaml"
    )
    assert manifest_path.exists()