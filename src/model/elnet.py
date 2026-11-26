"""ELNet model: a small CNN trained from scratch with multi-slice normalisation and BlurPool.

As in Tsai et al. (2020) and the authors' implementation (github.com/mxtsai/ELNet): every slice of one
series goes through the same 2D network. Channels 4K, 8K, 16K, 16K, 16K; blocks of convolution,
normalisation and ReLU added to their input (skip connections); anti-aliased downsampling (BlurPool,
5x5 binomial filter, stride 2). Each slice gives a 16K vector after spatial max-pooling, the maximum over
slices gives one vector per series, and a linear layer gives one logit.

Multi-slice normalisation: the slices are the batch dimension, so each slice is normalised on its own.
"layer" = layer normalisation (GroupNorm with 1 group), "contrast" = contrast (instance) normalisation
(GroupNorm with one group per channel). The paper uses "layer" for ACL and abnormal (axial plane) and
"contrast" for meniscus (coronal plane).

Difference from the authors' code: one output and binary cross-entropy instead of two outputs and
cross-entropy (the same model for a yes/no label), so that ELNet and MRNet use the same loss and the
same rule for choosing the epoch.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F


def norm_layer(channels, kind):
    if kind == "layer":
        return nn.GroupNorm(1, channels)
    if kind == "contrast":
        return nn.GroupNorm(channels, channels)
    raise ValueError(f"unknown normalisation: {kind}")


def init_uniform(module):
    """Kaiming uniform initialisation (the paper reports uniform initialisation)."""
    nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))


class BlurPool(nn.Module):
    """Blur with a fixed 5x5 binomial filter, then keep every second pixel (no padding, as in the authors' code)."""

    def __init__(self, channels, stride=2):
        super().__init__()
        k = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])
        k = torch.outer(k, k) / 256.0
        self.register_buffer("kernel", k.expand(channels, 1, 5, 5).clone())
        self.stride, self.channels = stride, channels

    def forward(self, x):
        return F.conv2d(x, self.kernel.to(x.dtype), stride=self.stride, groups=self.channels)


def conv_block(channels, kernel_size, repeats, norm):
    layers = []
    for _ in range(repeats):
        conv = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2, bias=False)
        init_uniform(conv)
        layers += [conv, norm_layer(channels, norm), nn.ReLU()]
    return nn.Sequential(*layers)


class ELNet(nn.Module):
    def __init__(self, K=4, norm="layer", dropout=0.0):
        super().__init__()
        c = [4 * K, 8 * K, 16 * K, 16 * K, 16 * K]
        self.conv1 = nn.Conv2d(1, c[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = norm_layer(c[0], norm)
        self.down1 = BlurPool(c[0])
        self.block2 = conv_block(c[0], 5, 2, norm)
        self.conv2 = nn.Conv2d(c[0], c[1], 5, padding=2, bias=False)
        self.down2 = BlurPool(c[1])
        self.block3 = conv_block(c[1], 3, 2, norm)
        self.conv3 = nn.Conv2d(c[1], c[2], 3, padding=1, bias=False)
        self.down3 = BlurPool(c[2])
        self.block4 = conv_block(c[2], 3, 1, norm)
        self.conv4 = nn.Conv2d(c[2], c[3], 3, padding=1, bias=False)
        self.down4 = BlurPool(c[3])
        self.block5 = conv_block(c[3], 3, 1, norm)
        self.conv5 = nn.Conv2d(c[3], c[4], 3, padding=1, bias=False)
        self.down5 = BlurPool(c[4])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(c[4], 1)
        for m in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.classifier]:
            init_uniform(m)

    def series_features(self, x):
        """x: one series, shape (slices, 1, H, W) or (1, slices, 1, H, W). Returns a (1, 16K) vector."""
        if x.dim() == 5:
            if x.shape[0] != 1:
                raise ValueError("one exam (one series) per batch")
            x = x[0]
        x = self.down1(F.relu(self.norm1(self.conv1(x))))
        x = x + self.block2(x)
        x = self.down2(F.relu(self.conv2(x)))
        x = x + self.block3(x)
        x = self.down3(F.relu(self.conv3(x)))
        x = x + self.block4(x)
        x = self.down4(F.relu(self.conv4(x)))
        x = x + self.block5(x)
        x = self.down5(F.relu(self.conv5(x)))
        f = self.dropout(torch.amax(x, dim=(2, 3)))  # (slices, 16K): spatial max-pooling
        return f.max(dim=0, keepdim=True).values  # (1, 16K): maximum over slices

    def forward(self, x):
        return self.classifier(self.series_features(x))  # (1, 1) logit


def count_parameters(model):
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    for norm in ["layer", "contrast"]:
        print(f"ELNet K=4 ({norm}) trainable parameters: {count_parameters(ELNet(4, norm)):,}")
