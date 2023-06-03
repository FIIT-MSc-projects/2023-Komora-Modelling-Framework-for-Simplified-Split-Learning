from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Union

import torch
import torch.nn as nn
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
    Parameters,
    ndarray_to_bytes,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
import numpy as np
Scalar = Union[bool, bytes, float, int, str]

import flwr as fl
from stage import Stage

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
            activations = net(inputs)
            server_inputs = activations.detach().clone()
            server_inputs = Variable(server_inputs, requires_grad=True) 

def test(net, testloader):
    return 0.0, 0.0


# Explain nn. Module and explain the forward and backward pass
class ResNet18Client(nn.Module):
    def __init__(self, config):
        super(ResNet18Client, self).__init__()
        self.cut_layer = config["cut_layer"]

        self.model = models.resnet18(pretrained=False)
        self.model = nn.ModuleList(self.model.children())
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        for i, l in enumerate(self.model):
            if i > self.cut_layer:
                break
            x = l(x)
        return x
    

config = {"cut_layer": 3, "logits": 10}
net = ResNet18Client(config).to(DEVICE)
trainloader, testloader, num_examples = load_data()

def get_parameters(net):
    print("getting params")
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters):
    print("AKOZE setting params")
    # params_dict = zip(net.state_dict().keys(), parameters)
    # state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    # net.load_state_dict(state_dict, strict=False)
    
class CustomClient(fl.client.Client):

    def _forward(self):
        data = next(iter(trainloader))
        inputs = data[0].to(DEVICE)

        self.optimizer.zero_grad() 
        self.activations = self.net(inputs)
        server_inputs = self.activations.detach().clone()
        self.server_inputs = Variable(server_inputs, requires_grad=True) 
      
        return ndarrays_to_parameters(server_inputs.numpy())

    def _backward(self):
        self.activations.backward(self.server_inputs.grad)
        self.optimizer.step()

    def __init__(self, cid, net, trainloader, valloader):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.epoch = 0
        self.optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)


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
            parameters=parameters
        )

    def fit(self, ins: FitIns) -> FitRes:
        print(f"[Client {self.cid}] fit, config: {ins.config}")

        if ins.config["stage"] == Stage.FORWARD.value:
            print("I should do forward prop")
            server_inputs = self._forward()
            status = Status(code=Code.OK, message="Success")
            print(type(server_inputs))
            return FitRes(
                status=status,
                parameters=server_inputs,
                num_examples=len(self.trainloader),
                metrics={}
            )
        elif ins.config["stage"] == Stage.BACKWARD.value:
            print("I should do backward")
            self._backward()
            status = Status(code=Code.OK, message="Success")
            parameters = Parameters(tensor_type="", tensors=[]),
            return FitRes(
                status=status,
                parameters=parameters,
                num_examples=len(self.trainloader),
                metrics={}
            )
        
        # Deserialize parameters to NumPy ndarray's
        parameters_original = ins.parameters
        ndarrays_original = parameters_to_ndarrays(parameters_original)

        # Update local model, train, get updated parameters
        set_parameters(self.net, ndarrays_original)
        train(self.net, self.trainloader, epochs=1)
        ndarrays_updated = get_parameters(self.net)

        # Serialize ndarray's into a Parameters object
        parameters_updated = ndarrays_to_parameters(ndarrays_updated)
        print(type(parameters_updated))

        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return FitRes(
            status=status,
            parameters=parameters_updated,
            num_examples=len(self.trainloader),
            metrics={}
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        print(f"[Client {self.cid}] evaluate, config: {ins.config}")

        # Deserialize parameters to NumPy ndarray's
        parameters_original = ins.parameters
        ndarrays_original = parameters_to_ndarrays(parameters_original)

        set_parameters(self.net, ndarrays_original)
        loss, accuracy = test(self.net, self.valloader)

        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return EvaluateRes(
            status=status,
            loss=float(loss),
            num_examples=len(self.valloader),
            metrics={"accuracy": float(accuracy)},
        )

import datetime
cid = str(datetime.datetime.now())[-1]
fl.client.start_client(server_address='127.0.0.1:3000', client=CustomClient(cid, net=net, trainloader=trainloader, valloader=testloader))