import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Input: 3 x 240 x 320
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # After pool 1: 60 x 80
        # After pool 2: 30 x 40
        self.fc1 = nn.Linear(32 * 30 * 40, 128)
        self.fc2 = nn.Linear(128, 2) # 2 classes: daninha, nao_daninha

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 30 * 40)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
