from __future__ import annotations

import argparse
import logging

from .checkpointing import atomic_write_yaml, build_run_paths, checkpoint_schedule, save_checkpoint
from .env import require_env
from .manifest import RunManifest, get_current_git_commit


def main(argv=None) -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Dummy AMR-FMA experiment that only writes manifest.yaml"
    )
    parser.add_argument("--phase", default="P1")
    parser.add_argument("--model-family", default="llama3")
    parser.add_argument("--domain", default="medical")
    parser.add_argument("--fma-method", default="lora_sft")
    parser.add_argument("--base-model-id", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id", default="0001")
    parser.add_argument("--experiment-name", default="dummy_p1_smoke")
    parser.add_argument("--total-steps", type=int, default=1000)
    parser.add_argument("--num-checkpoints", type=int, default=5)
    parser.add_argument(
        "--dataset", default=None, help="Dataset on which the experiment is run (optional)"
    )

    args = parser.parse_args(argv)

    base_dir = require_env("BASE_OUTPUT_DIR")

    paths = build_run_paths(
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
        base_model_id=args.base_model_id,
        seed=args.seed,
        run_id=args.run_id,
        experiment_name=args.experiment_name,
        git_commit=get_current_git_commit(),
        dataset=args.dataset,
        hyperparams={
            "total_steps": args.total_steps,
            "num_checkpoints": args.num_checkpoints,
        },
    )

    atomic_write_yaml(paths.manifest_path, manifest.to_dict())

    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    paths.eval_dir.mkdir(parents=True, exist_ok=True)
    paths.activations_dir.mkdir(parents=True, exist_ok=True)

    dummy_artifact_path = run_dir / "dummy_artifact.txt"
    dummy_artifact_path.write_text("dummy checkpoint artifact\n", encoding="utf-8")

    schedule = checkpoint_schedule(args.total_steps, args.num_checkpoints)
    schedule_set = set(schedule)
    for step in range(args.total_steps):
        if step in schedule_set:
            save_checkpoint(
                paths, step, dummy_artifact_path, metadata={"source": "dummy_experiment"}
            )

    logging.info(f"Base output dir: {base_dir}")
    logging.info(f"Created dummy experiment at {run_dir}")
    logging.info(f"Wrote manifest to {paths.manifest_path}")


if __name__ == "__main__":
    main()
