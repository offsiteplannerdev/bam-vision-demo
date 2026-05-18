import numpy as np

from src.transforms import get_train_transforms, get_val_transforms


def test_train_transforms_output_shape() -> None:
    """Train transforms should return a CHW tensor at the configured size."""
    img_size = 224
    image = np.zeros((320, 480, 3), dtype=np.uint8)

    transformed = get_train_transforms(img_size)(image=image)

    assert transformed["image"].shape == (3, img_size, img_size)


def test_val_transforms_output_shape() -> None:
    """Validation transforms should return a CHW tensor at the configured size."""
    img_size = 224
    image = np.zeros((300, 300, 3), dtype=np.uint8)

    transformed = get_val_transforms(img_size)(image=image)

    assert transformed["image"].shape == (3, img_size, img_size)
