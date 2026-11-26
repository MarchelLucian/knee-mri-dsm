"""MRNet model: AlexNet features on each slice, global average pooling, max over slices and one output.

As in Bien et al. (2018): the convolutional part of AlexNet (pre-trained on ImageNet) is applied to
every slice of one series, each slice gives a 256-value vector after global average pooling, the
element-wise maximum over slices gives one vector per series, and a linear layer gives one logit.
"""

from torch import nn
from torchvision.models import AlexNet_Weights, alexnet


class MRNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
        self.features = alexnet(weights=weights).features  # only the convolutional part
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, 1)

    def series_features(self, x):
        """x: one series, shape (slices, 3, H, W) or (1, slices, 3, H, W). Returns a (1, 256) vector."""
        if x.dim() == 5:
            if x.shape[0] != 1:
                raise ValueError("one exam (one series) per batch")
            x = x[0]
        f = self.gap(self.features(x)).flatten(1)  # (slices, 256)
        return f.max(dim=0, keepdim=True).values  # (1, 256)

    def forward(self, x):
        return self.classifier(self.series_features(x))  # (1, 1) logit


def count_parameters(model):
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print(f"MRNet trainable parameters (one network): {count_parameters(MRNet(pretrained=False)):,}")
