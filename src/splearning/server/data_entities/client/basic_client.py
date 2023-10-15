import logging
import sys

import torch.distributed.rpc
import torch
import torch.nn as nn
from torch.distributed.rpc import RRef
import torch.distributed.autograd as dist_autograd
from torch.distributed.optim import DistributedOptimizer
import logging
import os
from collections import Counter
from copy import deepcopy

from splearning.server.model_deserialization import deserialize_model
from splearning.utils.data_structures import AbstractClient, ClientArguments


class BasicClient(AbstractClient):

    def __init__(self, client_id, args: ClientArguments):
        self.client_id = client_id

        self.epochs = args.get_epochs()
        self.start_logger()

        self.server_ref = args.get_server_ref()
        self.server_model_refs = args.get_server_model_refs()

        try:
            self.logger.info(f"Loading {os.getenv('client_model_1_path')}")
            self.input_model = deserialize_model(os.getenv('client_model_1_path'))
            self.logger.info(f"Loading {os.getenv('client_model_2_path')}")
            self.output_model = deserialize_model(os.getenv('client_model_2_path'))
        except:
            self.logger.error("Client models not found")

        self.criterion = nn.CrossEntropyLoss()

        lr = float(os.getenv("lr", 0.001))
        momentum = float(os.getenv("momentum", 0.9))

        print(f"lr: {lr}")
        print(f"momentum: {momentum}")

        self.dist_optimizer=  DistributedOptimizer(
                    torch.optim.SGD,
                    list(map(lambda x: RRef(x),self.output_model.parameters())) +  self.server_model_refs +  list(map(lambda x: RRef(x),self.input_model.parameters())),
                    lr=lr,
                    momentum=momentum
                )

        self.load_data()

    def update_model(self,last_alice_rref,last_alice_id):
        self.logger.info(f"Transfering weights from Alice{last_alice_id} to Alice{self.client_id}")
        model1_weights,model2_weights = last_alice_rref.rpc_sync().give_weights()
        self.input_model.load_state_dict(model1_weights)
        self.output_model.load_state_dict(model2_weights)

    def __forward(self, inputs, labels):
        # Input client model
        input_model_activation = self.input_model(inputs) 

        # Server model
        server_model_activation = self.server_ref.rpc_sync().train(input_model_activation)

        # Output client model
        output_model_activation = self.output_model(server_model_activation)

        loss = self.criterion(output_model_activation,labels)
        return loss

    def __backward(self, context_id, loss):
        dist_autograd.backward(context_id, [loss])
        self.dist_optimizer.step(context_id)

    def train(self):
        self.logger.info("Training")

        for _ in range(self.epochs):
            for inputs, labels in self.train_dataloader:
                with dist_autograd.context() as ctx_id:
                    loss = self.__forward(inputs, labels)
                    self.__backward(ctx_id, loss)

    def give_weights(self):
        return [deepcopy(self.input_model.state_dict()), deepcopy(self.output_model.state_dict())]

    def eval(self):
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in self.test_dataloader:
                images, labels = data
                # calculate outputs by running images through the network
                activation_alice1 = self.input_model(images)
                activation_bob = self.server_ref.rpc_sync().train(activation_alice1)  # model(activation_alice1)
                outputs = self.output_model(activation_bob)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        self.logger.info(f"Alice{self.client_id} Evaluating Data: {round(correct / total, 3)}")
        return correct, total

    def load_data(self):

        datapath = os.getenv("datapath")
        print(f"datapath: {datapath}")
        self.train_dataloader = torch.load(os.path.join(datapath ,f"data_worker{self.client_id}_train.pt"))
        self.test_dataloader = torch.load(os.path.join(datapath ,f"data_worker{self.client_id}_test.pt"))

        self.n_train = len(self.train_dataloader.dataset)
        self.logger.info("Local Data Statistics:")
        self.logger.info("Dataset Size: {:.2f}".format(self.n_train))
        self.logger.info(dict(Counter(self.test_dataloader.dataset[:][1].numpy().tolist())))

    def start_logger(self):

        self.logger = logging.getLogger(f"alice{self.client_id}")
        self.logger.setLevel(logging.INFO)

        format = logging.Formatter("%(asctime)s: %(message)s")
        log_file_path = os.getenv('log_file')

        if not os.path.isdir(log_file_path):
            os.makedirs(log_file_path, exist_ok=True)

        fh = logging.FileHandler(filename=f"{log_file_path}/alice{self.client_id}.log",mode='w')
        fh.setFormatter(format)
        fh.setLevel(logging.INFO)

        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(format)
        sh.setLevel(logging.DEBUG)

        self.logger.addHandler(fh)
        self.logger.addHandler(sh)

        self.logger.info("Alice is going insane!")

