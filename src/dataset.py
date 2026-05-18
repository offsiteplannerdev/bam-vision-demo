from pathlib import Path
from typing import Any

import cv2
import torch
from torch.utils.data import Dataset


class CarPartDefectDataset(Dataset[tuple[torch.Tensor, int]]):
    """Dataset for car part defect classification images.

    Expects a folder layout with one subdirectory per class:

    ```text
    root_dir/
      defect/
      ok/
    ```

    Attributes:
        class_to_idx: Mapping from class name to integer label.
        samples: List of image paths and labels.
    """

    class_to_idx = {"ok": 0, "defect": 1}
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def __init__(self, root_dir: Path, transforms: Any | None = None) -> None:
        """Initializes the dataset.

        Args:
            root_dir: Directory containing `defect` and `ok` class folders.
            transforms: Optional albumentations transform pipeline.
        """
        self.root_dir = Path(root_dir)
        self.transforms = transforms
        self.samples = self._discover_samples()

        if not self.samples:
            raise ValueError(f"No images found under {self.root_dir}")

    def __len__(self) -> int:
        """Returns the number of available image samples."""
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Loads one sample and applies configured transforms.

        Args:
            index: Sample index.

        Returns:
            Tuple of image tensor and integer class label.
        """
        image_path, label = self.samples[index]
        # TODO: handle corrupted images gracefully
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            transformed = self.transforms(image=image)
            image = transformed["image"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image, label

    def _discover_samples(self) -> list[tuple[Path, int]]:
        """Discovers image files under known class directories.

        Returns:
            Sorted list of `(path, label)` tuples.
        """
        samples: list[tuple[Path, int]] = []
        for class_name, label in self.class_to_idx.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue

            for image_path in sorted(class_dir.rglob("*")):
                if image_path.is_file() and image_path.suffix.lower() in self.image_extensions:
                    samples.append((image_path, label))

        return samples
