import random

import numpy as np
import torch

from src.utils import seed_everything, seed_worker


def test_seed_everything_reseeds_common_generators() -> None:
    """Seeding should reset Python, NumPy, and PyTorch RNG streams."""
    seed_everything(123)
    first_values = (random.random(), float(np.random.rand()), float(torch.rand(1).item()))

    seed_everything(123)
    second_values = (random.random(), float(np.random.rand()), float(torch.rand(1).item()))

    assert first_values == second_values
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_seed_worker_reseeds_python_and_numpy() -> None:
    """DataLoader worker seeding should make Python and NumPy deterministic."""
    torch.manual_seed(123)
    seed_worker(0)
    first_values = (random.random(), int(np.random.randint(0, 10_000)))

    torch.manual_seed(123)
    seed_worker(0)
    second_values = (random.random(), int(np.random.randint(0, 10_000)))

    assert first_values == second_values
