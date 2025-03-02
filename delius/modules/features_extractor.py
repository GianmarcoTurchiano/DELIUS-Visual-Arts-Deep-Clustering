import torch
from torch import nn
from torchvision.models import densenet121, DenseNet121_Weights


class DenseNetFeaturesExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        densenet = densenet121(weights=DenseNet121_Weights.DEFAULT)
        self.features = densenet.features
        self.pooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.features(x)
        x = self.pooling(x)
        return torch.flatten(x, 1)
