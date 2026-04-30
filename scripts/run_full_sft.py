"""Entry point for full-parameter SFT training.

Usage:
    python scripts/run_full_sft.py run.run_id=my_run run.experiment_name=my_exp
    python scripts/run_full_sft.py model=tinygpt2 runtime=cpu run.run_id=smoke run.experiment_name=smoke
"""

from __future__ import annotations

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from amr_fma.fma.full_sft import train
from amr_fma.fma.training_config import TrainingConfig

load_dotenv()


@hydra.main(version_base=None, config_path="../configs/", config_name="config_full_sft")
def main(cfg: DictConfig) -> None:
    raw: dict = OmegaConf.to_object(cfg)  # type: ignore[assignment]
    train(TrainingConfig.from_dict(raw))


if __name__ == "__main__":
    main()
