import torch
from torch import nn


class DefectClassifier(nn.Module):
    """Simple CNN classifier for defect vs no-defect images."""

    def __init__(self, num_classes: int = 2, dropout: float = 0.25) -> None:
        """Initializes the CNN baseline.

        Args:
            num_classes: Number of output classes.
            dropout: Dropout probability used before the final classifier.
        """
        super().__init__()
        # NOTE: consider replacing with pretrained ResNet18 for production
        self.features = nn.Sequential(
            self._conv_block(3, 32),
            self._conv_block(32, 64),
            self._conv_block(64, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Runs a forward pass.

        Args:
            inputs: Batch tensor shaped `(N, 3, H, W)`.

        Returns:
            Class logits shaped `(N, num_classes)`.
        """
        features = self.features(inputs)
        pooled = self.pool(features)
        return self.classifier(pooled)

    @staticmethod
    def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
        """Creates a convolutional block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.

        Returns:
            Convolution, normalization, activation, and pooling block.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
