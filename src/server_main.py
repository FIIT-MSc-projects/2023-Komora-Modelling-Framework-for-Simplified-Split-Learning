import argparse
import os
from server.server import start_server
from models.models import *
from utils.data_structures import dotdict
from dotenv import load_dotenv

if __name__ == "__main__":

    load_dotenv(dotenv_path="server.env")

    parser = argparse.ArgumentParser(description='Split Learning Initialization')
    parser.add_argument('--world_size',type=int,default=3,help='The world size which is equal to 1 server + (world size - 1) clients')
    parser.add_argument('--port',type=str,default="9005",help='master port')
    parser.add_argument('--host',type=str,default="localhost",help='master hostname')    
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
        'datapath': '../data',
        'partition_alpha': 0.5,
        'port': params.port,
        'host': params.host
    })

    args.client_num_in_total = args.world_size - 1

    world_size = args.world_size
    start_server(world_size, args)