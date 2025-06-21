import torch
from torch import nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, channels=24):
        super(Discriminator, self).__init__()
        
        # sub-network I
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Updated linear1: correctly sized for 1*64*64 feature maps = 4096 features
        self.linear1 = nn.Sequential(
            nn.Linear(1 * 64 * 64, 128),
            nn.Tanh()
        )
        self.linear2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh()
        )
        self.linear3 = nn.Sequential(
            nn.Linear(64, 3)
        )

    def forward(self, x):
        # x => (batch, 2, H, W)
        x = self.max_pool1(x)
        x = self.max_pool2(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        feature_maps = x

        # Generic flatten
        x = x.view(x.size(0), -1)

        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = nn.Softmax(dim=1)(x)
        return x, feature_maps
