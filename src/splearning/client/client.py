import os
import torch.distributed.rpc as rpc
import torch.distributed as dist
from torch.distributed import TCPStore, init_process_group

from splearning.utils.data_structures import StartClientArguments


def init_env(port, address):
    os.environ["MASTER_PORT"] = port
    os.environ["MASTER_ADDR"] = address
    os.environ["HYDRA_FULL_ERROR"]="1"

def start_client(args: StartClientArguments):

    print(f"Starting client {args.get_name()}{args.get_rank()}")
    init_env(args.get_port(), args.get_address())

    os.environ["RANK"] = str(args.get_rank())
    os.environ["WORLD_SIZE"] = str(args.get_world_size())

    print("ENVIRON: ", os.environ)

    # dist.init_process_group(rank=args.get_rank(), world_size=args.get_world_size(), init_method='tcp://147.175.145.55:8888')
    # options = rpc.RpcBackendOptions(init_method='env://', rpc_timeout=1000)
    # init_process_group(
    #     backend="gloo", init_method="env://"
    #     # backend="gloo", init_method="tcp://147.175.145.55:8888", rank=args.get_rank(), world_size=args.get_world_size()
    # )

    # print("WORLD_SIZE: ", args.get_world_size())
    rpc.init_rpc(
        name=f"{args.get_name()}{args.get_rank()}", 
        rank=args.get_rank(), 
        world_size=args.get_world_size(),
        backend=rpc.BackendType.TENSORPIPE,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
            num_worker_threads=8,
            rpc_timeout=20 # 20 second timeout
        )
    )

    # rpc.init_rpc(name=f"{args.get_name()}{args.get_rank()}", rank=args.get_rank(), world_size=args.get_world_size())#, rpc_backend_options=options)
    rpc.shutdown()