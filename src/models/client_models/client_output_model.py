import torch.nn as nn
import torch.nn.functional as F

class output_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.fc3(x)
        return x