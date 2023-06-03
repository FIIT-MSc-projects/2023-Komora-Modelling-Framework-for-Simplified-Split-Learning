
print("++++++++++++++++++++++++++++++=========\n\n\n\n\n\n\n\n\n\n\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n<<<<<<<<<<<<<<<<<<<<<<<<")

from SplitNNAPI import init_client
from SplitNNAPI import init_server
from SplitNNAPI import SplitNN_distributed
from client_manager import SplitNNClientManager
from server_manager import SplitNNServerManager
from client import SplitNN_client
from server import SplitNN_server

import torch
# from mpi4py import MPI
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor


# server_model = torch.nn.Sequential(torch.nn.Linear(512, 10))

# server_args = {
#     "max_rank": 1,
#     # "comm": MPI.COMM_NULL,
#     "model": server_model
# }

# print("++++++++++++++++++++++++++++++=========\n\n\n\n\n\n\n\n\n\n\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n<<<<<<<<<<<<<<<<<<<<<<<<")
# server = SplitNN_server()


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
client_model = torch.nn.Sequential(torch.nn.Linear(28*28, 512))
client_args = {
    "client_index": 1, 
    # "comm": MPI.COMM_SELF,
    "model": client_model,
    "train_loader": train_dataloader,
    "test_loder": test_dataloader,
    "rank": 1,
    "max_rank": 1,
    "epochs": 5,
    "server_rank": 0,
    "device": torch.device("cpu")
}

client = SplitNN_client(client_args)
client_manager = SplitNNClientManager(client_args, client, backend="GRPC")