import os
import torch.distributed.rpc as rpc


def init_env(port, address):
    print("Initialize client")
    os.environ["MASTER_PORT"] = port
    os.environ['MASTER_ADDR'] = address

def start_client(name,rank,world_size,port, address):
    init_env(port, address)

    rpc.init_rpc(f"{name}{rank}", rank=rank, world_size=world_size)
    rpc.shutdown()