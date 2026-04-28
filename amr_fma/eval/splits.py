"""Train/eval split helpers, reusable across SFT and DPO-style training."""

from __future__ import annotations

from datasets import Dataset

# Constant so the held-out slice is identical across training seeds in a phase,
# which keeps eval/loss curves directly comparable across runs.
DEFAULT_SPLIT_SEED = 0


def train_eval_split(
    dataset: Dataset,
    *,
    eval_samples: int | None,
    seed: int = DEFAULT_SPLIT_SEED,
) -> tuple[Dataset, Dataset | None]:
    """Hold out a fixed-size eval slice. Returns ``(dataset, None)`` when disabled."""

    if eval_samples is None:
        return dataset, None
    if eval_samples < 1:
        raise ValueError(f"eval_samples must be at least 1 or null, got {eval_samples}")
    if eval_samples >= len(dataset):
        raise ValueError(
            f"eval_samples ({eval_samples}) must be smaller than dataset size ({len(dataset)})"
        )

    splits = dataset.train_test_split(test_size=eval_samples, seed=seed)
    return splits["train"], splits["test"]
