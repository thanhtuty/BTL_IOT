"""Small Gabor + depthwise-separable CNN for 64x64 palm ROI images.

Training is performed on a PC. The inference operations are intentionally
simple enough for a standalone ESP32 float32 reference runtime.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

INPUT_SIDE = 64
EMBED_DIM = 128
PERSON_IDS = ("person_01", "person_02", "person_03", "person_04")


def make_gabor_bank(kernel_size=7, orientations=4, sigma=2.0, wavelength=4.0):
    """Fixed zero-mean, unit-L2 filters; no pretrained weights required."""
    xy = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
    yy, xx = torch.meshgrid(xy, xy, indexing="ij")
    bank = []
    for i in range(orientations):
        angle = math.pi * i / orientations
        xr = xx * math.cos(angle) + yy * math.sin(angle)
        yr = -xx * math.sin(angle) + yy * math.cos(angle)
        kernel = torch.exp(-(xr.square() + yr.square()) / (2 * sigma * sigma))
        kernel = kernel * torch.cos((2 * math.pi / wavelength) * xr)
        kernel = kernel - kernel.mean()
        kernel = kernel / kernel.square().sum().sqrt().clamp(min=1e-8)
        bank.append(kernel)
    return torch.stack(bank).unsqueeze(1)


class SeparableBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            input_channels, input_channels, kernel_size=3,
            stride=stride, padding=1, groups=input_channels, bias=False
        )
        self.pointwise = nn.Conv2d(input_channels, output_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(output_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(self.bn(self.pointwise(self.depthwise(x))))


class TinyPalmNet(nn.Module):
    """4 Gabor 7x7 -> stem -> 5 separable blocks -> 128D embedding.

    The classifier is used only during training; on-device authentication
    uses the normalized embedding and four enrolled centroids.
    """

    def __init__(self, num_people=4):
        super().__init__()
        self.register_buffer("gabor", make_gabor_bank())
        self.stem = nn.Sequential(
            nn.Conv2d(4, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList([
            SeparableBlock(16, 24, stride=1),  # 32 x 32
            SeparableBlock(24, 32, stride=2),  # 16 x 16
            SeparableBlock(32, 48, stride=2),  # 8 x 8
            SeparableBlock(48, 64, stride=2),  # 4 x 4
            SeparableBlock(64, 64, stride=1),  # 4 x 4
        ])
        self.embedding = nn.Linear(64, EMBED_DIM)
        self.classifier = nn.Linear(EMBED_DIM, num_people)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1:] != (1, INPUT_SIDE, INPUT_SIDE):
            raise ValueError("Expected B x 1 x 64 x 64 input")
        x = F.relu(F.conv2d(x, self.gabor, padding=3))
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        embedding = F.normalize(self.embedding(x), dim=1, eps=1e-12)
        logits = self.classifier(embedding)
        return embedding, logits
