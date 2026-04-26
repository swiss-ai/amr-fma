"""Entry point for LoRA SFT training.

Usage:
    python scripts/run_lora_sft.py --config-path ../configs/example --config-name pilot_tinygpt2
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import hydra
from omegaconf import DictConfig, OmegaConf

from amr_fma.fma.lora_sft import train
from amr_fma.fma.training_config import TrainingConfig


@hydra.main(version_base=None, config_path="../configs/", config_name="pilot_tinygpt2")
def main(config: DictConfig) -> None:
    train(TrainingConfig.from_dict(OmegaConf.to_object(config)))


if __name__ == "__main__":
    main()
