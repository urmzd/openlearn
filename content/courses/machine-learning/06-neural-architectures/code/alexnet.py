"""alexnet.py -- AlexNet (Krizhevsky et al., 2012) in PyTorch (README §5).

The canonical reference architecture: 5 conv layers + 3 FC layers, ~60M
parameters, 1000-way ImageNet softmax. The original was split across two
GTX 580 GPUs because it didn't fit in 3 GB; this is the single-stream version
that every framework ships today.

The six choices that made it win (README §5): ReLU (non-saturating), dropout
in the FC head, overlapping max-pooling, local response normalization (here
kept for fidelity to the paper; modern nets use BatchNorm instead), 2-GPU
training, and aggressive data augmentation.

Run (needs torch):
    pip install torch
    python alexnet.py        # prints layer-by-layer output shapes + param count
"""

from __future__ import annotations

import torch
from torch import nn


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        # Feature extractor: conv -> ReLU -> (LRN) -> overlapping maxpool.
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),  # 224 -> 55
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
            nn.MaxPool2d(kernel_size=3, stride=2),  # overlapping: 3x3 / stride 2
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # Classifier head: dropout guards the 60M-parameter FC layers.
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def main() -> None:
    model = AlexNet(num_classes=1000)
    params = sum(p.numel() for p in model.parameters())
    print(f"AlexNet parameters: {params:,}")  # ~62.4M

    # Trace one 224x224 RGB image through the feature stack to see the shapes.
    x = torch.zeros(1, 3, 224, 224)
    for layer in model.features:
        x = layer(x)
        if isinstance(layer, (nn.Conv2d, nn.MaxPool2d)):
            print(f"{layer.__class__.__name__:12s} -> {tuple(x.shape)}")
    logits = model(torch.zeros(1, 3, 224, 224))
    print("logits:", tuple(logits.shape))  # (1, 1000)


if __name__ == "__main__":
    main()
