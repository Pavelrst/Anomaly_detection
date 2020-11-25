import torch.nn as nn
import torch.nn.functional as F

class IdentityBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(IdentityBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.conv_residual = nn.Conv2d(in_channels, out_channels, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        residual = self.conv_residual(input)
        x = self.activation(x+residual)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2, stride=2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.conv_residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn_residual = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        residual = self.conv_residual(input)
        residual = self.bn_residual(residual)
        x = self.activation(x+residual)
        return x

class ConvGroup(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ConvGroup, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if stride == 2:
            self.block1 = ConvBlock(in_channels, out_channels, kernel_size)
        else:
            self.block1 = IdentityBlock(in_channels, out_channels, kernel_size)
        self.block2 = IdentityBlock(out_channels, out_channels, kernel_size)
        self.block3 = IdentityBlock(out_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


class Net(nn.Module):
    def __init__(self, num_of_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3)
        self.group1 = ConvGroup(in_channels=8, out_channels=16, kernel_size=3)
        self.group2 = ConvGroup(in_channels=16, out_channels=32, kernel_size=3, stride=2)
        self.group3 = ConvGroup(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        self.fc = nn.Linear(64*8*8, num_of_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = x.view(-1, 64*8*8)
        x = self.fc(x)
        return x
