import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(img_size: int) -> A.Compose:
    """Builds the training augmentation pipeline.

    Args:
        img_size: Square output image size in pixels.

    Returns:
        Albumentations composition for training images.
    """
    return A.Compose(
        [
            A.Resize(height=img_size, width=img_size),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.35),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def get_val_transforms(img_size: int) -> A.Compose:
    """Builds the validation preprocessing pipeline.

    Args:
        img_size: Square output image size in pixels.

    Returns:
        Albumentations composition for validation images.
    """
    return A.Compose(
        [
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
