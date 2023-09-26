import argparse
import os
from client.client import start_client
from load_data.mnist_flat_generator import load_mnist_image
from dotenv import load_dotenv
import pickle

from models.models import model1

if __name__ == "__main__":


    client_model_1 = model1()
    if not os.path.isfile("client_model_1"):
        with open("client_model_1", "wb") as f:
            pickle.dump(client_model_1, f)
    load_dotenv(dotenv_path="client.env")

    parser = argparse.ArgumentParser(description='Split Learning Initialization')
    parser.add_argument('--rank',type=int,default=1,help='Rank of worker')
    parser.add_argument('--name',type=str,default="alice",help='Name of worker')
    parser.add_argument('--datapath',type=str,default="../data",help='folder path to all the local datasets')
    parser.add_argument('--clients',type=int,default=2,help='The number of clients')
    parser.add_argument('--partition_alpha',type=float,default=0.5,help='Number to describe the uniformity during sampling (heterogenous data generation for LDA)')
    parser.add_argument('--batch_size',type=int,default=16,help='The batch size during the epoch training')
    parser.add_argument('--port',type=str,default="8888",help='master port')
    parser.add_argument('--host',type=str,default="localhost",help='master hostname')

    args = parser.parse_args()

    load_mnist_image(args)
    start_client(name="alice",rank=int(args.rank),world_size=int(args.clients)+1,port=args.port, address=args.host)