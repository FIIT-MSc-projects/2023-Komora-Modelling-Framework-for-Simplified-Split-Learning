import argparse
from server.server import start_server
from models.models import *
from utils.data_structures import dotdict


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Split Learning Initialization')
    parser.add_argument('--world_size',type=int,default=3,help='The world size which is equal to 1 server + (world size - 1) clients')
    parser.add_argument('--port',type=str,default="8888",help='master port')
    params = parser.parse_args()

    args = dotdict({
        'epochs': 1,
        'iterations': 10,
        'batch_size': 16,
        'lr': 0.001,
        'world_size': params.world_size,
        'client_model_1': model1, 
        'client_model_2': model3,
        'server_model': model2,
        'port': params.port
    })

    args.client_num_in_total = args.world_size - 1

    world_size = args.world_size
    start_server(world_size, args)