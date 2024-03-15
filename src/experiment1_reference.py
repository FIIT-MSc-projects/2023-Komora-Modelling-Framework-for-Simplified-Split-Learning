import logging
import sys
import time
import torch.nn as nn
import torch.optim as optim

from data_handling.data_handling_experiment_1.prepare_mnist_data import load_image_datasets, prepare_data
from data_handling.data_handling_experiment_2.prepare_cifar_data_split import get_dataset_memory_size
from models.experiment_references.experiment1.model import model
from splearning.utils.testing import simple_evaluate
from splearning.utils.training import simple_train

EPOCHS = 5
BATCH_SIZE = 16
LOG_FILE_PATH = "../experiment1/reference.log"
DATA_PATH = "../experiment1/data/"

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

train_dataset, test_dataset = load_image_datasets(DATA_PATH, shape=(28, 28))
train_dataloader, test_dataloader = prepare_data(train_dataset, test_dataset, BATCH_SIZE)

print(f"Training dataset size: {get_dataset_memory_size(train_dataset) / (1024*1024)} MB")
print(f"Testing dataset size: {get_dataset_memory_size(test_dataset) / (1024*1024)} MB")

logger.info(f"Training dataset size: {len(train_dataset)}")
logger.info(f"Testing dataset size: {len(test_dataset)}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(NET.parameters(), lr=0.001, momentum=0.9)

total_training_time = 0

for epoch in range(EPOCHS):
    NET.train()
    total_training_time += simple_train(optimizer, train_dataloader, NET, logger.info)
    NET.eval()
    correct, total = simple_evaluate(test_dataloader, NET, logger.info)

end_time = time.time()
logger.info(f"Total training time: {total_training_time}")


    
