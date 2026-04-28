"""Dataset loading and normalization helpers."""

from __future__ import annotations

from typing import Any

from datasets import Dataset, load_dataset

from amr_fma.fma.training_config import TrainingConfig

# Fixed independently of run.seed so eval metrics are comparable across training seeds.
_EVAL_SPLIT_SEED = 0


def load_dataset_for_sft(config: TrainingConfig) -> tuple[Dataset, Dataset | None]:
    """Load and normalize a dataset into a text-only format for SFTTrainer.

    Returns (train_dataset, eval_dataset), where eval_dataset is None when
    dataset.eval_samples is not set.
    """

    dataset_split = load_dataset(config.dataset.name, split=config.dataset.split)

    if config.dataset.max_samples is not None:
        max_rows = min(config.dataset.max_samples, len(dataset_split))
        dataset_split = dataset_split.select(range(max_rows))

    candidate_text_fields = [config.dataset.text_field, "text", "content"]
    text_field = next(
        (field for field in candidate_text_fields if field in dataset_split.column_names),
        None,
    )

    if text_field is not None:
        dataset_split = (
            dataset_split.rename_column(text_field, "text")
            if text_field != "text"
            else dataset_split
        )
    else:
        dataset_split = dataset_split.map(
            _format_example, remove_columns=dataset_split.column_names
        )

    if config.dataset.eval_samples is None:
        return dataset_split, None

    splits = dataset_split.train_test_split(
        test_size=config.dataset.eval_samples, seed=_EVAL_SPLIT_SEED
    )
    return splits["train"], splits["test"]


def _format_example(example: dict[str, Any]) -> dict[str, str]:
    if "messages" in example and isinstance(example["messages"], list):
        lines: list[str] = []
        for message in example["messages"]:
            role = str(message.get("role", "user")).strip().lower()
            content = str(message.get("content", "")).strip()
            if content:
                lines.append(f"{role}: {content}")
        return {"text": "\n".join(lines)}

    if {"instruction", "input", "output"}.issubset(example):
        instruction = str(example.get("instruction", "")).strip()
        input_text = str(example.get("input", "")).strip()
        output_text = str(example.get("output", "")).strip()
        prompt = f"Instruction: {instruction}"
        if input_text:
            prompt = f"{prompt}\nInput: {input_text}"
        return {"text": f"{prompt}\nResponse: {output_text}"}

    if {"prompt", "response"}.issubset(example):
        prompt = str(example.get("prompt", "")).strip()
        response = str(example.get("response", "")).strip()
        return {"text": f"User: {prompt}\nAssistant: {response}"}

    available_fields = ", ".join(sorted(example.keys()))
    raise ValueError(
        "Dataset row does not expose a supported schema. "
        "Expected one of: text/content/messages/instruction+input+output/prompt+response. "
        f"Found fields: {available_fields}"
    )
