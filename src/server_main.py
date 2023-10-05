import argparse
from splearning.server.server import start_server
from models.server_models.simple_server_model import simple_server_model
from splearning.server.data_entities.client import alice
from splearning.utils.data_structures import dotdict
from dotenv import load_dotenv

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Split Learning Initialization')
    parser.add_argument('--world_size',type=int,default=3,help='The world size which is equal to 1 server + (world size - 1) clients')
    parser.add_argument('--port',type=str,default="9005",help='master port')
    parser.add_argument('--host',type=str,default="localhost",help='master hostname')    
    parser.add_argument('--config',type=str,default="server.env",help='config file path')
    params = parser.parse_args()

    load_dotenv(dotenv_path=params.config)

    args = dotdict({
        'epochs': 1,
        'iterations': 10,
        'batch_size': 16,
        'lr': 0.001,
        'world_size': params.world_size,
        'server_model': simple_server_model,
        'datapath': '../data',
        'partition_alpha': 0.5,
        'port': params.port,
        'host': params.host,
        "config": params.config,
        'client': alice
    })

    args.client_num_in_total = args.world_size - 1

    world_size = args.world_size
    start_server(world_size, args)