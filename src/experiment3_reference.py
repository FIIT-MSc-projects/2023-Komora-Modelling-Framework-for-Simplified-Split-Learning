import logging
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from data_handling_experiment_3.prepare_cifar_data import load_image_datasets, prepare_data
from experiment3_resnet import ResNet
from splearning.utils.testing import simple_evaluate
from splearning.utils.training import simple_train

logger = logging.getLogger(f"reference")
logger.setLevel(logging.INFO)

format = logging.Formatter("%(asctime)s: %(message)s")
fh = logging.FileHandler(filename=f"../experiment3/reference_augment.log",mode='w')
fh.setFormatter(format)
fh.setLevel(logging.INFO)

sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(format)
sh.setLevel(logging.DEBUG)

logger.addHandler(fh)
logger.addHandler(sh)

logger.info("Reference is going insane!")

epochs = 20
    
res_net = ResNet()

datapath = "experiment3/data"
train_dataset, test_dataset = load_image_datasets(datapath)
train_dataloader, test_dataloader = prepare_data(train_dataset, test_dataset, 128)

logger.info(f"Training dataset size: {len(train_dataset)}")
logger.info(f"Testing dataset size: {len(test_dataset)}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(res_net.parameters(), lr=0.001, momentum=0.9)

total_training_time = 0

for epoch in range(epochs):
    res_net.train()
    total_training_time += simple_train(optimizer, train_dataloader, res_net, logger.info)
    res_net.eval()
    correct, total = simple_evaluate(test_dataloader, res_net, logger.info)

end_time = time.time()
logger.info(f"Total training time: {total_training_time}")


    
