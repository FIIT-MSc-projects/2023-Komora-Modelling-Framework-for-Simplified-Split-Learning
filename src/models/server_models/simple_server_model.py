import torch.nn as nn
import torch.nn.functional as F
import torch

class simple_server_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32*13*13,1000)
        self.fc2 = nn.Linear(1000,100)

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x