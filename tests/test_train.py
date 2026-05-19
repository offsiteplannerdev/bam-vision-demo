import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

from src.train import train_one_epoch


def test_train_one_epoch_does_not_step_epoch_scheduler_per_batch() -> None:
    """The epoch scheduler should remain unchanged until the caller steps it."""
    model = nn.Linear(2, 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=1.0)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    loader = [
        (torch.tensor([[1.0, 0.0]]), torch.tensor([0])),
        (torch.tensor([[0.0, 1.0]]), torch.tensor([1])),
    ]

    train_one_epoch(model, loader, criterion, optimizer, torch.device("cpu"))

    assert optimizer.param_groups[0]["lr"] == 1.0

    scheduler.step()

    assert optimizer.param_groups[0]["lr"] == 0.1
