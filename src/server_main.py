import argparse

from models.server_models.resnet_server_model import ResNet
from models.server_models.simple_server_model import server_model_cifar, simple_server_model
from models.server_models.transformer_model import model
from splearning.server.data_entities.server.basic_server import BasicServer
from splearning.server.data_entities.client.basic_client import BasicClient
from splearning.server.data_entities.server.basic_strategy import BasicStrategy
from splearning.server.data_entities.server.client_2_client_initialization_strategy import Client2ClientInitializationStrategy
from splearning.server.server import start_server
from splearning.utils.data_structures import StartServerArguments

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Split Learning Initialization')
    parser.add_argument('--world_size',type=int,default=3,help='The world size which is equal to 1 server + (world size - 1) clients')
    parser.add_argument('--port',type=str,default="8888",help='master port')
    parser.add_argument('--host',type=str,default="localhost",help='master hostname')    
    parser.add_argument('--config',type=str,default="/home/miso/School/year5/DP/split_learning_framework/src/env_configs/server.env",help='config file path')
    parser.add_argument('--epochs',type=int,default=1,help='number of training epochs')
    params = parser.parse_args()

    clients = {
        1: {"name": "alice1", 
            "declaration": BasicClient, 
            "args": { 
                "epochs": 1
            }
        },
        2: {"name": "alice2", 
            "declaration": BasicClient, 
            "args": { 
                "epochs": 1
            }
        }
    }


    args = StartServerArguments(
        port=params.port,
        host=params.host,
        config=params.config,
        world_size=params.world_size,
        client_declaration=BasicClient,
        server=BasicServer,
        server_model=model,
        epochs=params.epochs,
        server_strategy=BasicStrategy,
        clients_configs=clients,
        parallel_training=False
    )

    start_server(args)