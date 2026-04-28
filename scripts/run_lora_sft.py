"""Entry point for LoRA SFT training.

Usage:
    python scripts/run_lora_sft.py                              # tiny-gpt2 smoke run
    python scripts/run_lora_sft.py --config-name simple_lora    # full Llama-3.1-8B run
    python scripts/run_lora_sft.py runtime.wandb=false          # disable W&B for local runs
"""

from __future__ import annotations

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from amr_fma.fma.lora_sft import train
from amr_fma.fma.training_config import TrainingConfig

load_dotenv()


@hydra.main(version_base=None, config_path="../configs/", config_name="pilot_tinygpt2")
def main(config: DictConfig) -> None:
    train(TrainingConfig.from_dict(OmegaConf.to_object(config)))


if __name__ == "__main__":
    main()
