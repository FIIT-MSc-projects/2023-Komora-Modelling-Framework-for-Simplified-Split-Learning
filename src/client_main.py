import argparse
from client.client import start_client
from load_data.mnist_flat_generator import load_mnist_image

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Split Learning Initialization')
    parser.add_argument('--rank',type=int,default=1,help='Rank of worker')
    parser.add_argument('--name',type=str,default="alice",help='Name of worker')
    parser.add_argument('--datapath',type=str,default="../data/mnist_flat",help='folder path to all the local datasets')
    parser.add_argument('--clients',type=int,default=2,help='The number of clients')
    parser.add_argument('--partition_alpha',type=float,default=0.5,help='Number to describe the uniformity during sampling (heterogenous data generation for LDA)')
    parser.add_argument('--batch_size',type=int,default=16,help='The batch size during the epoch training')
    parser.add_argument('--port',type=str,default="8888",help='master port')
    parser.add_argument('--host',type=str,default="localhost",help='master hostname')

    args = parser.parse_args()

    load_mnist_image(args)
    start_client(name="alice",rank=int(args.rank),world_size=int(args.clients)+1,port=args.port, address=args.host)