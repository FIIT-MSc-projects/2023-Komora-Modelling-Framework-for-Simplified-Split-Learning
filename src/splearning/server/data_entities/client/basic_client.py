
import time
import torch.distributed.rpc
import torch
import torch.nn as nn
from torch.distributed.rpc import RRef
import torch.distributed.autograd as dist_autograd
from torch.distributed.optim import DistributedOptimizer
import logging
import os
import sys
from copy import deepcopy

from torchinfo import summary
from models.experiment1.model import input_model
from splearning.client.deserialization import load_model_from_yaml
from splearning.utils.data_structures import AbstractClient, ClientArguments
from splearning.utils.logging import init_logging
from splearning.utils.testing import simple_evaluate


class BasicClient(AbstractClient):

    def __init__(self, client_id, args: ClientArguments):
        self.client_id = client_id

        self.epochs = args.get_epochs()
        self.start_logger()

        self.server_ref = args.get_server_ref()
        self.server_model_refs = args.get_server_model_refs()

        self.input_model = load_model_from_yaml(os.getenv("input_model"))
        self.output_model = load_model_from_yaml(os.getenv("output_model"))

        if self.input_model is None:
            raise ValueError("Input model not provided")
        
        
        if self.output_model is None:
            raise ValueError("Output model not provided")
   
        self.criterion = nn.CrossEntropyLoss()

        lr = float(os.getenv("lr", 0.001))
        momentum = float(os.getenv("momentum", 0.9))
        input_refs = list(map(lambda x: RRef(x),self.input_model.parameters()))
        output_refs = list(map(lambda x: RRef(x),self.output_model.parameters()))

        self.dist_optimizer=  DistributedOptimizer(
            optimizer_class=torch.optim.SGD,
            params_rref=[*output_refs, *self.server_model_refs, *input_refs],
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
        data_to_server = input_model_activation.element_size() * input_model_activation.numel()

        # Server model
        server_model_activation = self.server_ref.rpc_sync().train(input_model_activation)
        # Output client model

        data_to_client = server_model_activation.element_size() * server_model_activation.numel()
        output_model_activation = self.output_model(server_model_activation)

        loss = self.criterion(output_model_activation,labels)
        return output_model_activation, loss, data_to_server, data_to_client

    def __backward(self, context_id, loss):
        dist_autograd.backward(context_id, [loss], False)
        self.dist_optimizer.step(context_id)

    def train_epoch(self):
        self.logger.info("Training epoch")

        start_time = time.time()
        running_loss = 0.0
        correct = 0
        total = 0
        total_data_server = 0
        total_data_client = 0

        for inputs, labels in self.train_dataloader:
            with dist_autograd.context() as ctx_id:
                outputs, loss, data_to_server, data_to_client = self.__forward(inputs, labels)
                total_data_server += data_to_server
                total_data_client += data_to_client

                self.__backward(ctx_id, loss)

                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        end_time = time.time()
        epoch_training_time = end_time - start_time

        loss = running_loss / len(self.train_dataloader)
        accuracy = correct / total

        self.logger.info(f"Epoch training time: {epoch_training_time}")
        self.logger.info(f"Loss: {loss}")
        self.logger.info(f"Accuracy: {100 * accuracy:.2f}%")
        self.logger.info(f"Data transmitted to server: {total_data_server/(1024 * 1024)}MB")
        self.logger.info(f"Data transmitted to client: {total_data_client/(1024 * 1024)}MB")

    def set_train(self):
        self.logger.info("Training mode - ON")
        self.input_model.train()
        self.server_ref.rpc_sync().set_train()
        self.output_model.train()

    def set_eval(self):
        self.logger.info("Training mode - OFF")
        self.input_model.eval()
        self.server_ref.rpc_sync().set_eval()
        self.output_model.eval()

    def train(self):
        for _ in range(self.epochs):
            self.set_train()
            self.train_epoch()
            self.set_eval()
            self.eval()

    def train_batch(self):
        self.batch_number += 1

        try:
            inputs, labels = next(self.iter_dataloader)
        except StopIteration:
            self.batch_number = 0
            self.iter_dataloader = iter(self.train_dataloader)
            inputs, labels = next(self.iter_dataloader)
            
        with dist_autograd.context() as ctx_id:
            loss = self.__forward(inputs, labels)
            self.__backward(ctx_id, loss)

    def get_total_batches(self):
        return self.total_batches


    def give_weights(self):
        return [deepcopy(self.input_model.state_dict()), deepcopy(self.output_model.state_dict())]

    def eval(self):
        self.logger.info("Evaluation")

        def output_function(inputs):
            activation_alice1 = self.input_model(inputs)
            activation_bob = self.server_ref.rpc_sync().train(activation_alice1)  # model(activation_alice1)
            outputs = self.output_model(activation_bob)

            return outputs
        
        correct, total = simple_evaluate(self.test_dataloader, output_function, self.logger.info)
        return correct, total

    def load_data(self):

        self.train_dataloader = torch.load(os.path.join(os.getenv("datapath"), f"train_dataset_cifar_{self.client_id}.pt"))
        self.test_dataloader = torch.load(os.path.join(os.getenv("datapath"), f"test_dataset_cifar_{self.client_id}.pt"))
        self.iter_dataloader = iter(self.train_dataloader)
        self.batch_number = 0
        self.total_batches = len(self.train_dataloader)

        self.n_train = len(self.train_dataloader)
        self.logger.info("Local Data Statistics:")
        self.logger.info("Dataset Size: {:.2f}".format(self.n_train))

    def start_logger(self):

        log_file_path = os.getenv('log_file')
        self.logger = init_logging(logger_name=f"alice{self.client_id}", log_file_path=log_file_path)
        self.logger.info("Alice is going insane!")

