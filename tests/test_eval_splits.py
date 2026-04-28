from __future__ import annotations

import pytest
from datasets import Dataset

from amr_fma.eval.splits import train_eval_split


def _dataset(size: int) -> Dataset:
    return Dataset.from_dict({"text": [f"row-{i}" for i in range(size)]})


def test_returns_full_dataset_when_eval_samples_none() -> None:
    dataset = _dataset(10)
    train, evaluation = train_eval_split(dataset, eval_samples=None)
    assert evaluation is None
    assert len(train) == 10


def test_split_sizes_and_disjoint() -> None:
    dataset = _dataset(100)
    train, evaluation = train_eval_split(dataset, eval_samples=20)
    assert evaluation is not None
    assert len(train) == 80
    assert len(evaluation) == 20
    assert set(train["text"]).isdisjoint(set(evaluation["text"]))


def test_split_is_deterministic_across_calls() -> None:
    dataset = _dataset(100)
    _, eval_first = train_eval_split(dataset, eval_samples=20)
    _, eval_second = train_eval_split(dataset, eval_samples=20)
    assert eval_first["text"] == eval_second["text"]


def test_eval_samples_must_be_smaller_than_dataset() -> None:
    with pytest.raises(ValueError, match="smaller than dataset size"):
        train_eval_split(_dataset(5), eval_samples=5)


def test_eval_samples_must_be_positive() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        train_eval_split(_dataset(5), eval_samples=0)
