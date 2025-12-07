import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """A small CNN for binary image classification.

    Input: 3xHxW (we resize to 240x320 in transforms)
    Output: logits for 2 classes
    """
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        # Make the network robust to input size by using AdaptiveAvgPool2d
        self.gap = nn.AdaptiveAvgPool2d((4, 4))
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_model(num_classes: int = 2) -> nn.Module:
    return SimpleCNN(num_classes=num_classes)
