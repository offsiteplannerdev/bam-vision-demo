import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from src.dataset import CarPartDefectDataset
from src.model import DefectClassifier
from src.transforms import get_train_transforms, get_val_transforms
from src.utils import load_config, seed_everything, seed_worker, setup_logger


def create_run_dir(output_dir: Path) -> Path:
    """Creates a unique timestamped run directory.

    Args:
        output_dir: Parent directory for training runs.

    Returns:
        Newly-created run directory.
    """
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = 1
    run_dir = output_dir / timestamp

    while True:
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_dir
        except FileExistsError:
            run_dir = output_dir / f"{timestamp}-{suffix:02d}"
            suffix += 1


def create_data_loader_generator(seed: int) -> torch.Generator:
    """Creates a seeded torch generator for DataLoader randomness.

    Args:
        seed: Seed used by PyTorch when shuffling and spawning workers.

    Returns:
        Seeded PyTorch generator.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def build_dataloaders(config: dict[str, Any], project_root: Path) -> tuple[DataLoader, DataLoader]:
    """Builds train and validation data loaders.

    Args:
        config: Parsed experiment configuration.
        project_root: Repository root used to resolve relative paths.

    Returns:
        Train and validation data loaders.
    """
    data_config = config["data"]
    training_config = config["training"]
    img_size = int(data_config["img_size"])
    data_seed = int(data_config.get("seed", config["project"]["seed"]))

    train_dir = project_root / Path(data_config["train_dir"])
    val_dir = project_root / Path(data_config["val_dir"])

    train_dataset = CarPartDefectDataset(train_dir, transforms=get_train_transforms(img_size))
    val_dataset = CarPartDefectDataset(val_dir, transforms=get_val_transforms(img_size))

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training_config["batch_size"]),
        shuffle=True,
        num_workers=int(data_config["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=create_data_loader_generator(data_seed),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(training_config["batch_size"]),
        shuffle=False,
        num_workers=int(data_config["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=create_data_loader_generator(data_seed + 1),
    )
    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Runs one training epoch.

    Args:
        model: Model being trained.
        loader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Target device.

    Returns:
        Average loss and accuracy for the epoch.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

    if total == 0:
        raise ValueError("Training loader did not yield any samples")

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluates the model on a validation loader.

    Args:
        model: Model being evaluated.
        loader: Validation data loader.
        criterion: Loss function.
        device: Target device.

    Returns:
        Average loss and accuracy.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

    if total == 0:
        raise ValueError("Validation loader did not yield any samples")

    return total_loss / total, correct / total


def train(config_path: Path) -> None:
    """Runs model training from a YAML config.

    Args:
        config_path: Path to experiment config.
    """
    config_path = Path(config_path).expanduser().resolve()
    project_root = config_path.parents[1]
    config = load_config(config_path)

    seed_everything(int(config["project"]["seed"]))
    logger = setup_logger("bam_vision.train")

    output_dir = project_root / Path(config["project"]["output_dir"])
    run_dir = create_run_dir(output_dir)
    logger.info("Loaded config from %s", config_path)
    logger.info("Experiment config:\n%s", yaml.safe_dump(config, sort_keys=True))
    logger.info("Writing run artifacts to %s", run_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    train_loader, val_loader = build_dataloaders(config, project_root)
    model = DefectClassifier(
        num_classes=int(config["model"]["num_classes"]),
        dropout=float(config["model"]["dropout"]),
    ).to(device)

    training_config = config["training"]
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    scheduler = StepLR(
        optimizer,
        step_size=int(training_config["scheduler_step_size"]),
        gamma=float(training_config["scheduler_gamma"]),
    )

    best_val_acc = float("-inf")
    for epoch in range(1, int(training_config["epochs"]) + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        logger.info(
            "epoch=%03d train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f lr=%.6f",
            epoch,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            optimizer.param_groups[0]["lr"],
        )
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = run_dir / f"best_model_epoch_{epoch:03d}.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                },
                checkpoint_path,
            )
            logger.info("Saved checkpoint to %s", checkpoint_path)


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Train BAM defect classifier")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML experiment config.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    logging.captureWarnings(True)
    args = parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
