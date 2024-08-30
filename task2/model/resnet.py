import torch
import torch.nn.functional as F

from torch import nn
from torchvision import models

MODELS = {
    "resnet18": (models.resnet18, 512),
    "resnet34": (models.resnet34, 512),
    "resnet50": (models.resnet50, 2048),
    "resnet101": (models.resnet101, 2048),
    "resnet152": (models.resnet152, 2048),
    "resnext50": (models.resnext50_32x4d, 2048),
    "resnext101": (models.resnext101_32x8d, 2048),
    "wideresnet50": (models.wide_resnet50_2, 2048),
    "wideresnet101": (models.wide_resnet101_2, 2048),
}

class GrayResNet(nn.Module):
    def __init__(self, in_channels=1, n_class=1, model_type="resnet50", weights=False, norm=nn.BatchNorm2d, **kwargs):
        super(GrayResNet, self).__init__()

        self.create_function, self.input_features = MODELS[model_type]
        self.model = self.create_function(weights=weights, norm_layer=norm, **kwargs)

        if in_channels != 3:
            if weights:
                with torch.no_grad():  # Temporarily disables gradient tracking
                    averaged_weights = self.model.conv1.weight.mean(dim=1, keepdim=True)
                    if self.model.conv1.bias is not None:
                        averaged_bias = self.model.conv1.bias  #
            
            self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

            if weights: # RGB to Grayscale Weights
                with torch.no_grad():
                    self.model.conv1.weight = nn.Parameter(averaged_weights)
                    if self.model.conv1.bias is not None:
                        self.model.conv1.bias = nn.Parameter(averaged_bias)

        self.model.avgpool = nn.AdaptiveAvgPool2d(1)
        
        self.model.fc = nn.Linear(self.input_features, n_class)
        self.model.fc.weight.data.normal_(mean=0.0, std=0.01)
        self.model.fc.bias.data.zero_()

        if not weights: nn.init.kaiming_normal_(self.model.conv1.weight, mode='fan_out', nonlinearity='relu')      

    def forward(self, x):
        return self.model(x)