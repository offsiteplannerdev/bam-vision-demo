from pathlib import Path

import torch

from src.train import create_data_loader_generator, create_run_dir


def test_create_run_dir_does_not_reuse_existing_directory(tmp_path: Path) -> None:
    """Run directories should be unique to avoid overwriting checkpoints."""
    first_run_dir = create_run_dir(tmp_path)
    second_run_dir = create_run_dir(tmp_path)

    assert first_run_dir.exists()
    assert second_run_dir.exists()
    assert first_run_dir != second_run_dir


def test_create_data_loader_generator_is_deterministic() -> None:
    """DataLoader generators should produce repeatable random sequences."""
    first_generator = create_data_loader_generator(42)
    second_generator = create_data_loader_generator(42)

    first_values = torch.rand(3, generator=first_generator)
    second_values = torch.rand(3, generator=second_generator)

    assert torch.equal(first_values, second_values)
