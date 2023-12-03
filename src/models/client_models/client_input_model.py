import torch
import torch.nn as nn
import torch.nn.functional as F

class input_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # x = x.view(-1,1,28,28)
        # x = self.conv1(x)
        # x = self.pool(F.relu(x))
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        return x
    
class input_model_cifar(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x