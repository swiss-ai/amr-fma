from __future__ import annotations

import pytest
import torch
from transformers.trainer_utils import EvalPrediction

from amr_fma.fma.lora_sft import compute_eval_metrics, preprocess_logits_for_metrics


def test_compute_eval_metrics_uniform_logits() -> None:
    vocab_size = 4
    # Uniform logits → every token equally likely → loss = log(vocab_size), perplexity = vocab_size.
    # argmax of uniform logits is token 0, which matches the labels below → accuracy = 1.0.
    logits = torch.zeros((2, 3, vocab_size))
    labels = torch.tensor([[0, 0, -100], [0, -100, 0]])

    # Simulate what the Trainer does: preprocess_logits_for_metrics runs on GPU during eval,
    # then the Trainer converts the result to numpy before passing to compute_eval_metrics.
    preprocessed = preprocess_logits_for_metrics(logits, labels).numpy()
    labels_np = labels.numpy()

    eval_pred = EvalPrediction(predictions=preprocessed, label_ids=labels_np)
    metrics = compute_eval_metrics(eval_pred)

    assert metrics["perplexity"] == pytest.approx(float(vocab_size))
    assert metrics["token_accuracy"] == pytest.approx(1.0)
    assert metrics["loss_std"] == pytest.approx(0.0)
