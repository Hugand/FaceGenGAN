import torch
from torch import nn
from torch.autograd import Variable


class Discriminator(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.conv0_layer = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),
            nn.ReLU())
        self.conv1_layer = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU())

        self.conv2_layer = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU())
        self.conv3_layer = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            #nn.MaxPool2d(3), # For now remove this max pooling to keep the 16x16 img size
            nn.ReLU())

        self.conv4_layer = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU())
        self.conv5_layer = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU())
        self.conv6_layer = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())

        # Linear layers
        self.linear0_layer = nn.Sequential(
            nn.Linear(in_features=16*16*256, out_features=4096),
            nn.ReLU())
        self.linear1_layer = nn.Sequential(
            nn.Linear(in_features=4096, out_features=2048),
            nn.Dropout(p=0.3),
            nn.ReLU())
        self.linear2_layer = nn.Sequential(
            nn.Linear(in_features=2048, out_features=2048),
            nn.Dropout(p=0.3),
            nn.ReLU())
        self.linear3_layer = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1),
            #nn.Sigmoid()
        )
    
    def __conv_head(self, inp):
        out = self.conv0_layer(inp)
        out = self.conv1_layer(out)
        out = self.conv2_layer(out)
        out = self.conv3_layer(out)
        out = self.conv4_layer(out)
        out = self.conv5_layer(out)
        out = self.conv6_layer(out)

        return out

    def __linear_classifier(self, inp):
        out = self.linear0_layer(inp)
        out = self.linear1_layer(out)
        out = self.linear2_layer(out)
        out = self.linear3_layer(out)

        return out

    def forward(self, features):
        out = self.__conv_head(features)
        flattened = out.view(out.size(0), -1)
        out = self.__linear_classifier(flattened)

        return out