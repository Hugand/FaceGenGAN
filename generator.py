import torch
from torch import nn
from torch.autograd import Variable


class Generator(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.conv0_layer = nn.Sequential(
            nn.ConvTranspose2d(32*32, 1024, 4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True))
            
        self.conv2_layer = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True))

        self.conv3_layer = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True))
            
        self.conv5_layer = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True))

        self.conv6_layer = nn.Sequential(
             nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True))

        self.conv7_layer = nn.Sequential(
            nn.ConvTranspose2d(128, 1, 3, stride=1, padding=1, bias=False),
            nn.Sigmoid())

    def forward(self, features):
        out = self.conv0_layer(features)
        out = self.conv2_layer(out)
        out = self.conv3_layer(out)
        out = self.conv5_layer(out)
        out = self.conv6_layer(out)
        reconstructed = self.conv7_layer(out)

        return reconstructed.view(-1, 1, 32, 32)
