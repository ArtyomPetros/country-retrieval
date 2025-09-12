import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class Embedder(nn.Module):
    def __init__(self, embed_dim: int=256, pretrained: bool=False):
        super().__init__()
        m = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        )
        self.backbone = m.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        in_ch = 576
        self.head = nn.Sequential(
            nn.BatchNorm1d(in_ch),
            nn.Dropout(0.2),
            nn.Linear(in_ch, embed_dim),
        )
    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x).flatten(1)
        x = self.head(x)
        x = F.normalize(x, dim=1)
        return x
