import timm.models.crossvit
import torch
import torch.nn as nn


class YCXNet(nn.Module):
    def __init__(self):
        super(YCXNet, self).__init__()
        self.model = timm.create_model('resnet18', pretrained=True, num_classes=176, in_chans=3)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, inputs):
        output = self.softmax(self.model(inputs))
        return output
