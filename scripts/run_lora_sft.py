"""Entry point for LoRA SFT training.

Usage:
    python scripts/run_lora_sft.py run.run_id=my_run run.experiment_name=my_exp
    python scripts/run_lora_sft.py model=tinygpt2 runtime=cpu run.run_id=smoke run.experiment_name=smoke
    python scripts/run_lora_sft.py model=llama3_8b dataset=chatdoctor lora.r=8 run.run_id=r8_run run.experiment_name=lora_r8
"""

from __future__ import annotations

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from amr_fma.fma.lora_sft import train
from amr_fma.fma.training_config import TrainingConfig

load_dotenv()


@hydra.main(version_base=None, config_path="../configs/", config_name="config")
def main(cfg: DictConfig) -> None:
    raw: dict = OmegaConf.to_object(cfg)  # type: ignore[assignment]
    train(TrainingConfig.from_dict(raw))


if __name__ == "__main__":
    main()
