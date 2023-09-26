from .data_entities import bob
import torch.distributed.rpc as rpc
import os

def init_env(port):
    print("Initialize Meetup Spot")
    os.environ['MASTER_ADDR'] = "localhost"
    os.environ["MASTER_PORT"] = port

def start_server(world_size,args):
    init_env(args.port)
    rpc.init_rpc("bob", rank=0, world_size=world_size)

    BOB = bob(args)

    for _ in range(args.iterations):
        for client_id in range(1,world_size):
            BOB.train_request(client_id)
        BOB.eval_request()

    rpc.shutdown()