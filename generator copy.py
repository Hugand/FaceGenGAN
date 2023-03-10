import torch
from torch import nn
from torch.autograd import Variable


class Generator(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.conv0_layer = nn.Sequential(
            nn.Conv2d(1, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(2),
            nn.ReLU())

        # Big residual block
        self.conv1_layer = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.conv2_layer = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())

        # Small residual block 1
        self.conv3_layer = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())

        # Small residual block 2
        self.conv4_layer = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        
        # Small residual block 3
        self.conv5_layer = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())

        # Small residual block 4
        self.conv9_layer = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())

        # Small residual block 5
        self.conv10_layer = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            #nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ReLU())


        self.conv6_layer = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ReLU())
        self.conv7_layer = nn.Sequential(
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.conv8_layer = nn.Sequential(
            nn.Conv2d(128, 1, 3, stride=1, padding=1),
            nn.ReLU())
        
        # self.output_layer = nn.Linear(
        #     in_features=32*32*8, out_features=kwargs["output_shape"]
        # )
    
    def __head(self, inp):
        residual = inp
        out = self.conv1_layer(inp)
        out = self.conv2_layer(out) * residual
        return out

    def __residual_batches(self, inp):
        out = self.conv3_layer(inp)
        big_res = out
        res = out
        out = self.conv4_layer(out) * res
        res = out
        out = self.conv5_layer(out) * res
        res = out
        out = self.conv9_layer(out) * res
        res = out
        out = self.conv10_layer(out) * res * big_res

        return out

    def __output_construction(self, inp):
        out = self.conv6_layer(inp)
        out = self.conv7_layer(out)
        out = self.conv8_layer(out)
        
        return out

    def forward(self, features):
        out = self.conv0_layer(features)

        out = self.__head(out)
        out = self.__residual_batches(out)
        reconstructed = self.__output_construction(out)

        return reconstructed.view(-1, 1, 32, 32)