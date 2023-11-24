import os
import argparse
from dotenv import load_dotenv
from data_handling_experiment_1.prepare_mnist_data_split import prepare_data, load_image_datasets

from splearning.client.client import start_client
from data_handling.mnist_flat_generator import load_mnist_image
from models.client_models.client_input_model import input_model, input_model2
from models.client_models.client_output_model import output_model, output_model2
from splearning.utils.data_structures import StartClientArguments, dotdict

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Split Learning Initialization')
    parser.add_argument('--rank',type=int,default=1,help='Rank of worker')
    parser.add_argument('--name',type=str,default="alice",help='Name of worker')
    parser.add_argument('--clients',type=int,default=2,help='The number of clients')
    parser.add_argument('--partition_alpha',type=float,default=0.5,help='Number to describe the uniformity during sampling (heterogenous data generation for LDA)')
    parser.add_argument('--batch_size',type=int,default=16,help='The batch size during the epoch training')
    parser.add_argument('--port',type=str,default="8888",help='master port')
    parser.add_argument('--host',type=str,default="localhost",help='master hostname')
    parser.add_argument('--config',type=str,default="client.env",help='config file path')

    args = parser.parse_args()

    load_dotenv(dotenv_path=args.config)

    data_args = dotdict({
        'clients': args.clients,
        'partition_alpha': args.partition_alpha,
        'datapath': os.getenv("datapath"),
        'batch_size': args.batch_size,
    })

    client_args = StartClientArguments(
        rank=int(args.rank),
        name=args.name,
        world_size=int(args.clients)+1,
        port=args.port,
        address=args.host,
        input_model=input_model,
        output_model=output_model
    )

    # load_mnist_image(data_args)
    train_dataset, test_dataset = load_image_datasets(os.getenv("datapath"), shape=(28,28), rank=int(args.rank), clients_total=int(args.clients))
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Testing dataset size: {len(test_dataset)}")
    prepare_data(train_dataset, test_dataset, int(args.clients), int(args.rank), os.getenv("datapath"))
    start_client(client_args)