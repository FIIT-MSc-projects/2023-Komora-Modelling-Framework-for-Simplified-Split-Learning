from splearning.utils.data_structures import AbstractServer, ServerArguments, ServerArguments, StartServerArguments
import torch.distributed.rpc as rpc
import os

from dotenv import load_dotenv

def init_env(port, address):
    print("Initialize Meetup Spot")
    os.environ["MASTER_PORT"] = port
    os.environ['MASTER_ADDR'] = address


def server_lifecycle(server: AbstractServer, world_size):

    iterations = os.getenv("iterations", 3)

    for _ in range(iterations):
        for client_id in range(1, world_size):
            server.train_request(client_id)
        server.eval_request()

def start_server(args: StartServerArguments):
    
    world_size = args.get_world_size()
    load_dotenv(args.get_config())
    
    init_env(port=args.get_port(), address=args.get_host())
    rpc.init_rpc("bob", rank=0, world_size=world_size)

    server_args = ServerArguments(
        client_num_in_total=args.get_client_num_in_total(),
        client_declaration=args.get_client_declaration(),
        epochs=args.get_epochs(),
        server_model=args.get_server_model(),
        server_strategy=args.get_server_strategy(),
        clients_configs=args.get_clients_configs()
    )

    server = (args.get_server())(server_args)
    server_lifecycle(server, world_size)

    rpc.shutdown()