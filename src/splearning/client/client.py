import os
import torch.distributed.rpc as rpc

from splearning.client.model_serialization import load_model_from_yaml, serialize_model
from splearning.utils.data_structures import StartClientArguments


def init_env(port, address):
    os.environ["MASTER_PORT"] = port
    os.environ['MASTER_ADDR'] = address

def serialize_models(input_model, output_model):
    print(os.getenv("client_model_1_path"))
    serialize_model(input_model, os.getenv("client_model_1_path"))
    serialize_model(output_model, os.getenv("client_model_2_path"))

def start_client(args: StartClientArguments):

    print(f"Starting client {args.get_name()}{args.get_rank()}")
    init_env(args.get_port(), args.get_address())
    serialize_models(args.get_input_model(), args.get_output_model())
    # input_model = load_model_from_yaml(os.getenv("input_model"), "input_model")
    # output_model = load_model_from_yaml(os.getenv("output_model"), "output_model")
    # serialize_models(input_model, output_model)

    rpc.init_rpc(f"{args.get_name()}{args.get_rank()}", rank=args.get_rank(), world_size=args.get_world_size())
    rpc.shutdown()