import torch
import torchvision
import torchvision. transforms as transforms
import torch.nn as nn
import torch.nn. functional as F
from torchvision import models 
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
transform = transforms.Compose(
[transforms. ToTensor(),
transforms. Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# CIFAR10 is a dataset of natural images consisting of 50k training images and 10k test # Every image is labelled with one of the following class
classes = ('plane', 'car', 'bird', 'cat',
'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
trainset = torchvision.datasets. CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader:DataLoader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testset = torchvision.datasets. CIFAR10 (root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)


# Explain nn. Module and explain the forward and backward pass
class ResNet18Client(nn.Module):
    """docstring for ResNet"""
    # Explain initialize (listing the neural network architecture and other related parameters) 
    def __init__(self, config):
        super(ResNet18Client, self).__init__()
        # Explain this line.
        self.cut_layer = config["cut_layer"]
        # Explain this line

        self.model = models.resnet18(pretrained=False)
        self.model = nn.ModuleList(self.model.children())
        self.model = nn.Sequential(*self.model)

    # Explain forward (actually used during the execution of the neural network at runtime) 
    def forward(self, x):
        for i, l in enumerate(self.model):
            if i > self.cut_layer:
                break
            x = l(x)
        return x
    
class ResNet18Server(nn.Module): 
    """docstring for ResNet"""
    def __init__(self, config):
        super(ResNet18Server, self).__init__()
        self.logits = config["logits"]
        self.cut_layer = config[ "cut_layer"]
        self.model = models.resnet18(pretrained=False) 
        num_ftrs = self.model.fc.in_features

        # Explain this part
        self.model.fc = nn.Sequential(nn.Flatten(), nn.Linear(num_ftrs, self.logits))
        self.model = nn.ModuleList(self.model.children()) 
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        for i, l in enumerate(self.model): 
            # Explain this part
            if i <= self.cut_layer: 
                continue
            x = l(x)
        return nn.functional.softmax(x, dim=1)
    
config = {"cut_layer": 3, "logits": 10}
client_model = ResNet18Client(config).to(device)
server_model = ResNet18Server(config).to(device)

criterion = nn.CrossEntropyLoss()
client_optimizer = optim.SGD(client_model.parameters(), lr=0.01, momentum=0.9)
server_optimizer = optim.SGD(server_model.parameters(), lr=0.01, momentum=0.9)

num_epochs = 50
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        client_optimizer.zero_grad() 
        server_optimizer.zero_grad()

        # Client part
        activations = client_model(inputs)
        server_inputs = activations.detach().clone()

        # Simulation of server part is happening in this portion 
        # # Server part
        server_inputs = Variable(server_inputs, requires_grad=True) 
        outputs = server_model(server_inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # server optimization
        server_optimizer.step()

        # Simulation of Client Happening in this portion
        # Client optimization
        activations.backward(server_inputs.grad)
        client_optimizer.step()

        running_loss += loss.item()

        if i % 200 == 199:
            print('[{}, {}] loss: {}'.format(epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0