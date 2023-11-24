import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from data_handling_experiment_1.prepare_mnist_data import load_image_datasets, prepare_data
from splearning.utils.testing import simple_evaluate
from splearning.utils.training import simple_train

epochs = 5

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
    
simple_conv_net = model()

datapath = "experiment1/data"
train_dataset, test_dataset = load_image_datasets(datapath, shape=(28,28))
train_dataloader, test_dataloader = prepare_data(train_dataset, test_dataset, 16)

print(f"Training dataset size: {len(train_dataset)}")
print(f"Testing dataset size: {len(test_dataset)}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(simple_conv_net.parameters(), lr=0.001, momentum=0.9)

total_training_time = 0

for epoch in range(epochs):
    simple_conv_net.train()
    total_training_time += simple_train(optimizer, train_dataloader, simple_conv_net)
    simple_conv_net.eval()
    correct, total = simple_evaluate(test_dataloader, simple_conv_net)

end_time = time.time()
print(f"Total training time: {total_training_time}")


    
