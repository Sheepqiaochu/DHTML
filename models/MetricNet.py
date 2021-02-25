import torch
import torch.nn as nn
from models.base import FaceModel


class MetricNet(FaceModel):
    FEATURE_DIMENSION = 512

    def __init__(self, n_classes):
        super().__init__(num_classes=n_classes, feature_dim=self.FEATURE_DIMENSION)
        # build layers
        self.layer1 = nn.Linear(256, 512)
        self.layer2 = nn.ReLU(inplace=True)
        self.layer3 = nn.Linear(512, n_classes) if self.num_classes else None

    def forward(self, x):
        x = self.layer1(x)
        features = self.layer2(x)
        logits = self.layer3(x) if self.num_classes else None
        feature_normed = features.div(
            torch.norm(features, p=2, dim=1, keepdim=True).expand_as(features))
        return logits, feature_normed

