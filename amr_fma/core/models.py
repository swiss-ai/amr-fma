from __future__ import annotations

import logging
from pathlib import Path
from typing import Any


class ModelNotFoundError(RuntimeError):
    """Raised when a Hugging Face model identifier cannot be resolved."""


_TOKENIZER_CACHE: dict[str, Any] = {}


def load_base_model(
    model_id: str,
    torch_dtype: Any = None,
    device_map: str = "auto",
    trust_remote_code: bool = False,
) -> Any:
    """Load a causal language model and cache its tokenizer."""

    try:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("transformers is required to load models") from exc

    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    except Exception as exc:
        raise ModelNotFoundError(f"Unable to resolve model identifier: {model_id}") from exc

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            device_map=device_map,
            torch_dtype=torch_dtype,
            attn_implementation="sdpa",
            trust_remote_code=trust_remote_code,
        )
    except Exception as exc:
        raise ModelNotFoundError(f"Unable to load model: {model_id}") from exc

    _TOKENIZER_CACHE[model_id] = tokenizer
    try:
        model._amr_fma_tokenizer = tokenizer
    except Exception:
        logging.debug("Unable to attach tokenizer to model instance for %s", model_id)
    return model


def prepare_lora_model(model: Any, lora_config: Any) -> Any:
    """Wrap a base model in a PEFT LoRA adapter."""

    try:
        from peft import get_peft_model
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("peft is required to prepare LoRA models") from exc

    return get_peft_model(model, lora_config)


def save_lora_adapter(model: Any, save_path: Path) -> Path:
    """Save a PEFT adapter to disk and return the save path."""

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("torch is required to save LoRA adapters") from exc

    with torch.no_grad():
        model.save_pretrained(save_path)

    return save_path
