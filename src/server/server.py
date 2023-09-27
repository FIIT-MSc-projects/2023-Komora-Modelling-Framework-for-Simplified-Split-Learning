from .data_entities.simple_server import bob
import torch.distributed.rpc as rpc
import os

def init_env(port, address):
    print("Initialize Meetup Spot")
    os.environ["MASTER_PORT"] = port
    os.environ['MASTER_ADDR'] = address


def start_server(world_size,args):
    init_env(args.port, args.host)
    rpc.init_rpc("bob", rank=0, world_size=world_size)

    BOB = bob(args)

    for _ in range(args.iterations):
        for client_id in range(1,world_size):
            BOB.train_request(client_id)
        BOB.eval_request()

    rpc.shutdown()