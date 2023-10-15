import argparse
from splearning.server.data_entities.server.basic_server import BasicServer, Client2ClientInitializationStrategy
from splearning.server.data_entities.client.basic_client import BasicClient
from splearning.server.start_server import start_server
from models.server_models.simple_server_model import simple_server_model
from splearning.utils.data_structures import ServerArguments, StartServerArguments

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Split Learning Initialization')
    parser.add_argument('--world_size',type=int,default=3,help='The world size which is equal to 1 server + (world size - 1) clients')
    parser.add_argument('--port',type=str,default="9005",help='master port')
    parser.add_argument('--host',type=str,default="localhost",help='master hostname')    
    parser.add_argument('--config',type=str,default="server.env",help='config file path')
    parser.add_argument('--epochs',type=int,default=1,help='number of training epochs')
    params = parser.parse_args()

    args = StartServerArguments(
        port=params.port,
        host=params.host,
        config=params.config,
        world_size=params.world_size,
        client=BasicClient,
        server=BasicServer,
        server_model=simple_server_model,
        epochs=params.epochs,
        server_strategy=Client2ClientInitializationStrategy
    )

    start_server(args)