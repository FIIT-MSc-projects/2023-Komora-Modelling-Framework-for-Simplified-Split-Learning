from collections import OrderedDict

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import models 
from torch.autograd import Variable
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

import flwr as fl
from stage import Stage

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def load_data():
    """Load CIFAR-10 (training and test set)."""
    transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = CIFAR10(".", train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
    testset = CIFAR10(".", train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=128, shuffle=False)
    num_examples = {"trainset" : len(trainset), "testset" : len(testset)}
    return trainloader, testloader, num_examples

def train(net, trainloader, epochs):
    """Train the network on the training set."""

    client_optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)

            client_optimizer.zero_grad() 
            # server_optimizer.zero_grad()

            # Client part
            activations = net(inputs)
            server_inputs = activations.detach().clone()

            # Simulation of server part is happening in this portion 
            # # Server part
            server_inputs = Variable(server_inputs, requires_grad=True) 


            ######################################
            # outputs = server_model(server_inputs)
            # loss = criterion(outputs, labels)
            # loss.backward()

            # # server optimization
            # server_optimizer.step()

            # # Simulation of Client Happening in this portion
            # # Client optimization
            # activations.backward(server_inputs.grad)
            # client_optimizer.step()

            # running_loss += loss.item()

            # if i % 200 == 199:
            #     print('[{}, {}] loss: {}'.format(epoch + 1, i + 1, running_loss / 200))
            #     running_loss = 0.0

def backward(net, trainloader, epochs):
    pass

def test(net, testloader):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy


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
    

config = {"cut_layer": 3, "logits": 10}
# Load model and data
net = ResNet18Client(config).to(DEVICE)
trainloader, testloader, num_examples = load_data()


class CifarClient(fl.client.NumPyClient):
    def __init__(self):
        super().__init__()
        self.stage = Stage.RECORD_ALIGNMENT

    def get_parameters(self, config):
        print("Getting parameters")
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        print("Setting parameters")
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        print("Fitting")
        if True or self.stage == Stage.FORWARD:
            self.set_parameters(parameters)
            train(net, trainloader, epochs=1)
            return self.get_parameters(config={}), num_examples["trainset"], {}
        if self.stage == Stage.BACKWARD:
            pass


    def evaluate(self, parameters, config):
        print("Evaluating")
        self.set_parameters(parameters)
        #loss, accuracy = test(net, testloader)
        # return float(loss), num_examples["testset"], {"accuracy": float(accuracy)}
        return 0.0, 5, {0.0}
    
def get_parameters(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
    
class CustomClient(fl.client.Client):

    def __init__(self, cid, net, trainloader, valloader):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        print(f"[Client {self.cid}] get_parameters")

        # Get parameters as a list of NumPy ndarray's
        ndarrays = get_parameters(self.net)

        # Serialize ndarray's into a Parameters object
        parameters = ndarrays_to_parameters(ndarrays)

        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return GetParametersRes(
            status=status,
            parameters=parameters,
        )

    def fit(self, ins: FitIns) -> FitRes:
        print(f"[Client {self.cid}] fit, config: {ins.config}")

        # Deserialize parameters to NumPy ndarray's
        parameters_original = ins.parameters
        ndarrays_original = parameters_to_ndarrays(parameters_original)

        # Update local model, train, get updated parameters
        set_parameters(self.net, ndarrays_original)
        train(self.net, self.trainloader, epochs=1)
        ndarrays_updated = get_parameters(self.net)

        # Serialize ndarray's into a Parameters object
        parameters_updated = ndarrays_to_parameters(ndarrays_updated)

        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return FitRes(
            status=status,
            parameters=parameters_updated,
            num_examples=len(self.trainloader),
            metrics={},
        )


fl.client.start_numpy_client(server_address='127.0.0.1:8081', client=CifarClient())