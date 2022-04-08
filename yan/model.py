from importlib_metadata import re
import timm.models.crossvit
import torch
import torch.nn as nn


class YCXNet(nn.Module):
    # def __init__(self):
    #     super(YCXNet, self).__init__()
    #     self.model = timm.create_model('resnet18', pretrained=True, num_classes=176, in_chans=3)
    #     self.softmax = torch.nn.Softmax(dim=1)

    # def forward(self, inputs):
    #     output = self.softmax(self.model(inputs))
    #     return output
    def __init__(self):
        super(YCXNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(10, 20, kernel_size=5), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(320, 120), nn.Sigmoid(),
            nn.Linear(120, 84), nn.Sigmoid(),
            nn.Linear(84, 10))

    def forward(self, X):
        # X=torch.rand(size=(1,3,28,28),dtype=torch.float32)
        output = self.net(X)
        return output
