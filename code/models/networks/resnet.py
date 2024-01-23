"""
This script is taken and adapted from the official SEN12MS repository:
https://github.com/schmitt-muc/SEN12MS/blob/master/
"""

import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision import models

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data)

def fc_init_weights(m):
    if type(m) == nn.Linear:
        init.kaiming_normal_(m.weight.data)

class ResNet50(nn.Module):
    def __init__(self, n_inputs = 12, num_classes = 17):
        super().__init__()

        resnet = models.resnet50(pretrained=False)

        self.conv1 = nn.Conv2d(n_inputs, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.encoder = nn.Sequential(
            self.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )
        self.FC = nn.Linear(2048, num_classes)

        self.apply(weights_init_kaiming)
        self.apply(fc_init_weights)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        logits = self.FC(x)

        return logits