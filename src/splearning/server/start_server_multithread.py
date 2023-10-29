import torch
import torch.multiprocessing as mp
from splearning.utils.data_structures import AbstractServer, ServerArguments, ServerArguments, StartServerArguments
import torch.distributed.rpc as rpc

from threading import Thread


from dotenv import load_dotenv
import os


def init_env(port, address):
    print("Initialize Meetup Spot")
    os.environ["MASTER_PORT"] = port
    os.environ['MASTER_ADDR'] = address

def simple_rpc(id):
    print(os.getpid())
    rpc.rpc_sync(to=f"alice{id}",func=print, args=("HELLO",))
    print(f"DONE")

def simple_func():
    print(os.getpid())


def multithread_server_lifecycle(server: AbstractServer, iterations):
    # NOTE: this is required for the ``fork`` method to work

    for i in range(iterations):
        server.train_clients()
        server.eval_request()

    # processes = []

    # for rank in range(world_size-1):
    #     # p = mp.Process(target=server.train_request, args=(rank+1,))
    #     p = mp.Process(target=simple_rpc, args=(rank+1,))
    #     p.start()
    #     processes.append(p)

    # for p in processes:
    #     p.join()



def start_server_multithread(args: StartServerArguments):
    
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
    multithread_server_lifecycle(server, iterations=10)

    rpc.shutdown()