import os
import json
from dotenv import load_dotenv
import torch
from torch.distributed.optim import DistributedOptimizer

load_dotenv(dotenv_path="./env_configs/client1.env")

optimizer_params = json.loads(os.getenv("optimizer_params"))
optimizer_class_name = os.getenv("optimizer_name")
optimizer_class = getattr(torch.optim, optimizer_class_name)

optim = DistributedOptimizer(
    optimizer_class=optimizer_class,
    params_rref=[],
    **optimizer_params
)

criterion = getattr(torch.nn, os.getenv("loss"))()

print(optim)
print(criterion)
print(torch.nn.CrossEntropyLoss())
