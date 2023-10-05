import os
import torch.distributed.rpc as rpc


def init_env(port, address):
    os.environ["MASTER_PORT"] = port
    os.environ['MASTER_ADDR'] = address

def start_client(name,rank,world_size,port, address):
    print(f"Starting client {name}{rank}")
    init_env(port, address)
    rpc.init_rpc(f"{name}{rank}", rank=rank, world_size=world_size)
    rpc.shutdown()