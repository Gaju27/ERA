import torch
import torch.nn as nn
import torch.nn.functional as F


class ciferNet(nn.Module):
    def __init__(self) -> None:
        super(ciferNet, self).__init__()

        # Block 1: 32x32x3 -> 16x16x32
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1, stride=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )

        # Block 2: 16x16x32 -> 6x6x64
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, stride=1, dilation=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=2, padding=1, dilation=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )

        # Block 3: 6x6x64 -> 6x6x128
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=2, stride=1, dilation=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )

        # Block 4: 6x6x128 -> 6x6x256
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, stride=1, dilation=2, groups=128, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Conv2d(in_channels=256, out_channels=10, kernel_size=(1, 1), bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.gap(x)
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        return F.log_softmax(x, dim=-1)


