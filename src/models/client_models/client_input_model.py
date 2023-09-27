import torch.nn as nn
import torch.nn.functional as F

class input_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,32, 3)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # x = x.view(-1,1,28,28)
        x = self.pool(F.relu(self.conv1(x)))
        return x