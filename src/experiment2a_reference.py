import logging
import sys
import time
import torch.nn as nn
import torch.optim as optim

from data_handling.data_handling_experiment_2.prepare_cifar_data import load_image_datasets, prepare_data
from models.experiment2.model import model
from splearning.utils.testing import simple_evaluate
from splearning.utils.training import simple_train

EPOCHS = 10
BATCH_SIZE = 4
LOG_FILE_PATH = "../experiment2/a/reference.log"
DATA_PATH = "../experiment2/a/data/"

NET = model()

###############################

logger = logging.getLogger(f"reference")
logger.setLevel(logging.INFO)

format = logging.Formatter("%(asctime)s: %(message)s")
fh = logging.FileHandler(filename=LOG_FILE_PATH,mode='w')
fh.setFormatter(format)
fh.setLevel(logging.INFO)

sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(format)
sh.setLevel(logging.DEBUG)

logger.addHandler(fh)
logger.addHandler(sh)

train_dataset, test_dataset = load_image_datasets(DATA_PATH)
train_dataloader, test_dataloader = prepare_data(train_dataset, test_dataset, BATCH_SIZE)

logger.info(f"Training dataset size: {len(train_dataset)}")
logger.info(f"Testing dataset size: {len(test_dataset)}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(NET.parameters(), lr=0.001, momentum=0.9)

total_training_time = 0

for epoch in range(EPOCHS):
    model.train()
    total_training_time += simple_train(optimizer, train_dataloader, model, logger.info)
    model.eval()
    correct, total = simple_evaluate(test_dataloader, model, logger.info)

end_time = time.time()
logger.info(f"Total training time: {total_training_time}")


    
