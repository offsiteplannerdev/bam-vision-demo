import logging
import os
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
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse YAML config: {config_path}") from exc

    if not isinstance(config, dict):
        raise ValueError(f"Config did not parse to a dictionary: {config_path}")

    return config


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seeds common random number generators for repeatable experiments.

    Args:
        seed: Integer seed value.
        deterministic: Whether to request deterministic PyTorch kernels where possible.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    """Seeds random number generators in a PyTorch DataLoader worker.

    Args:
        worker_id: DataLoader worker identifier supplied by PyTorch.
    """
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


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

    logger.propagate = False

    return logger
