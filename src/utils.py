import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: Path) -> dict[str, Any]:
    """Loads a YAML configuration file.

    Args:
        path: Path to a YAML config file.

    Returns:
        Parsed configuration dictionary.
    """
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Config did not parse to a dictionary: {config_path}")

    return config


def seed_everything(seed: int) -> None:
    """Seeds common random number generators for repeatable experiments.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(name: str) -> logging.Logger:
    """Creates a console logger.

    Args:
        name: Logger name.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
