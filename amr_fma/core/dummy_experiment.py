# amr_fma/core/dummy_experiment.py
from __future__ import annotations

import argparse

from .env import require_env
from .manifest import RunManifest, get_current_git_commit
from .paths import RunPaths


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Dummy AMR-FMA experiment that only writes manifest.yaml"
    )
    parser.add_argument("--phase", default="P1")
    parser.add_argument("--model-family", default="llama3")
    parser.add_argument("--domain", default="medical")
    parser.add_argument("--fma-method", default="lora_sft")
    parser.add_argument("--base-model-id", default="meta-llama/Llama-3-8B-Instruct-GGUF")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id", default="0001")
    parser.add_argument("--experiment-name", default="dummy_p1_smoke")
    parser.add_argument(
        "--dataset", default=None, help="Dataset on which the experiment is run (optional)"
    )

    args = parser.parse_args(argv)

    BASE_DIR = require_env("BASE_OUTPUT_DIR")

    paths = RunPaths(
        phase=args.phase,
        model_family=args.model_family,
        domain=args.domain,
        fma_method=args.fma_method,
        seed=args.seed,
        run_id=args.run_id,
    )

    run_dir = paths.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = RunManifest(
        phase=args.phase,
        model_family=args.model_family,
        domain=args.domain,
        fma_method=args.fma_method,
        seed=args.seed,
        run_id=args.run_id,
        experiment_name=args.experiment_name,
        git_commit=get_current_git_commit(),
    )

    manifest_yaml = manifest.to_yaml()
    paths.manifest_path.write_text(manifest_yaml, encoding="utf-8")

    # Optionally create empty subdirs
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    paths.eval_dir.mkdir(parents=True, exist_ok=True)
    paths.activations_dir.mkdir(parents=True, exist_ok=True)

    print(f"Base output dir: {BASE_DIR}")
    print(f"Created dummy experiment at {run_dir}")
    print(f"Wrote manifest to {paths.manifest_path}")


if __name__ == "__main__":
    main()
