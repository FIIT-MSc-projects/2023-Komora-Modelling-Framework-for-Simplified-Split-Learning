import torch
import torch.nn as nn
import torch.nn.functional as F

class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32*13*13,1000)
        self.fc2 = nn.Linear(1000,100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1,1,28,28)
        x = self.conv1(x)
        x = self.pool(F.relu(x))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x