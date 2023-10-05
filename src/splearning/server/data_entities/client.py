import logging
import sys

import torch.distributed.rpc
import torch
import torch.nn as nn
from torch.distributed.rpc import RRef
import torch.distributed.rpc as rpc
import torch.distributed.autograd as dist_autograd
from torch.distributed.optim import DistributedOptimizer
import logging
import os
from collections import Counter
from copy import deepcopy

from ..model_deserialization import deserialize_model

class alice(object):

    def __init__(self,server,bob_model_rrefs,rank,args):
        self.client_id = rank
        self.epochs = args.epochs
        self.start_logger()

        self.bob = server

        try:
            self.logger.info(f"Loading {os.getenv('client_model_1_path')}")
            self.model1 = deserialize_model(os.getenv('client_model_1_path'))
            self.logger.info(f"Loading {os.getenv('client_model_2_path')}")
            self.model2 = deserialize_model(os.getenv('client_model_2_path'))
        except:
            self.logger.error("Client models not found")

        self.criterion = nn.CrossEntropyLoss()

        self.dist_optimizer=  DistributedOptimizer(
                    torch.optim.SGD,
                    list(map(lambda x: RRef(x),self.model2.parameters())) +  bob_model_rrefs +  list(map(lambda x: RRef(x),self.model1.parameters())),
                    lr=args.lr,
                    momentum = 0.9
                )

        self.load_data(args)

    def train(self,last_alice_rref,last_alice_id):
        self.logger.info("Training")

        if last_alice_rref is None:
            self.logger.info(f"Alice{self.client_id} is first client to train")

        else:
            self.logger.info(f"Alice{self.client_id} receiving weights from Alice{last_alice_id}")
            model1_weights,model2_weights = last_alice_rref.rpc_sync().give_weights()
            self.model1.load_state_dict(model1_weights)
            self.model2.load_state_dict(model2_weights)


        for epoch in range(self.epochs):
            for i,data in enumerate(self.train_dataloader):
                inputs,labels = data

                with dist_autograd.context() as context_id:

                    activation_alice1 = self.model1(inputs)
                    activation_bob = self.bob.rpc_sync().train(activation_alice1) #model(activation_alice1)
                    activation_alice2 = self.model2(activation_bob)

                    loss = self.criterion(activation_alice2,labels)

                    # run the backward pass
                    dist_autograd.backward(context_id, [loss])

                    self.dist_optimizer.step(context_id)

    def give_weights(self):
        return [deepcopy(self.model1.state_dict()), deepcopy(self.model2.state_dict())]

    def eval(self):
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in self.test_dataloader:
                images, labels = data
                # calculate outputs by running images through the network
                activation_alice1 = self.model1(images)
                activation_bob = self.bob.rpc_sync().train(activation_alice1)  # model(activation_alice1)
                outputs = self.model2(activation_bob)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        self.logger.info(f"Alice{self.client_id} Evaluating Data: {round(correct / total, 3)}")
        return correct, total

    def load_data(self,args):
        self.train_dataloader = torch.load(os.path.join(args.datapath ,f"data_worker{self.client_id}_train.pt"))
        self.test_dataloader = torch.load(os.path.join(args.datapath ,f"data_worker{self.client_id}_test.pt"))

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
            os.mkdir(log_file_path)

        fh = logging.FileHandler(filename=f"{log_file_path}/alice{self.client_id}.log",mode='w')
        fh.setFormatter(format)
        fh.setLevel(logging.INFO)

        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(format)
        sh.setLevel(logging.DEBUG)

        self.logger.addHandler(fh)
        self.logger.addHandler(sh)

        self.logger.info("Alice is going insane!")

