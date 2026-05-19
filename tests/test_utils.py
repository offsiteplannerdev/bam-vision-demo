import random

import numpy as np
import torch

from src.utils import seed_everything


def test_seed_everything_repeats_random_sequences() -> None:
    """Seeding should make Python, NumPy, and Torch draws repeatable."""
    seed_everything(123)
    python_value = random.random()
    numpy_value = np.random.random()
    torch_value = torch.rand(1)

    seed_everything(123)

    assert random.random() == python_value
    assert np.random.random() == numpy_value
    assert torch.equal(torch.rand(1), torch_value)


def test_seed_everything_requests_deterministic_torch_kernels() -> None:
    """Seeding should disable cuDNN autotuning and request deterministic kernels."""
    seed_everything(123)

    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    assert torch.are_deterministic_algorithms_enabled() is True
