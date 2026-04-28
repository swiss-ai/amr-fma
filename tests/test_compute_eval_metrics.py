from __future__ import annotations

import pytest
import torch
from transformers.trainer_utils import EvalPrediction

from amr_fma.fma.lora_sft import compute_eval_metrics


def test_compute_eval_metrics_uniform_logits() -> None:
    vocab_size = 4
    logits = torch.zeros((2, 3, vocab_size)).tolist()
    labels = torch.tensor([[0, 0, -100], [0, -100, 0]]).tolist()

    eval_pred = EvalPrediction(predictions=logits, label_ids=labels)
    metrics = compute_eval_metrics(eval_pred)

    assert metrics["perplexity"] == pytest.approx(float(vocab_size))
    assert metrics["token_accuracy"] == pytest.approx(1.0)
    assert metrics["loss_std"] == pytest.approx(0.0)
