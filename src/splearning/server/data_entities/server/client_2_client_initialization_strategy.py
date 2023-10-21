import os
import sys
import logging
import torch.distributed.rpc as rpc
from splearning.utils.data_structures import AbstractServerStrategy
from splearning.utils.data_structures import ClientArguments

class Client2ClientInitializationStrategy(AbstractServerStrategy):
    def __init__(self):
        self.__init_logger()
        self.last_alice_id = None

    def execute_train_request(self, clients, client_id):
        self.logger.info(f"Train Request for client with id: {client_id}")

        current_client = clients[client_id]

        if self.last_alice_id is not None:
            current_client.rpc_sync(timeout=0).update_model(clients[self.last_alice_id], self.last_alice_id)

        current_client.rpc_sync(timeout=0).train()
        self.last_alice_id = client_id

    def execute_eval_request(self, clients, total_client_num):
        self.logger.info("Initializing Evaluation of all clients")

        total = []
        num_corr = []
        check_eval = [clients[client_id].rpc_async(timeout=0).eval() for client_id in range(1, total_client_num + 1)]

        for check in check_eval:
            corr, tot = check.wait()
            total.append(tot)
            num_corr.append(corr)

        self.logger.info("Accuracy over all data: {:.3f}".format(sum(num_corr) / sum(total)))

    def create_clients_from_configs(self, clients_configs, server_ref, server_model_refs):
        self.logger.info("Starting the initialization of the array of clients")

        clients = {}

        for rank in clients_configs:
            client = clients_configs[rank]
            client_args = ClientArguments(server_ref, server_model_refs, client["args"]["epochs"])
            clients[rank] = rpc.remote(client["name"], client["declaration"], (rank, client_args))

        return clients

    def create_clients(self, client_declaration, client_args, total_client_number):
        self.logger.info("Starting the initialization of the array of clients")

        clients = {}
        client_name_pattern = os.getenv('client', default="alice*")

        for rank in range(total_client_number):
            self.logger.info(f"Initializing the client: {client_name_pattern.replace('*', str(rank+1))}")
            clients[rank + 1] = rpc.remote(client_name_pattern.replace("*", str(rank+1)), client_declaration, (rank+1, client_args))

        return clients

    def __init_logger(self):
        self.logger = logging.getLogger("Server")
        self.logger.setLevel(logging.INFO)

        format = logging.Formatter("%(asctime)s: %(message)s")
        log_file_path = os.getenv('log_file')

        print(log_file_path)

        if not os.path.isdir(log_file_path):
            os.makedirs(log_file_path, exist_ok=True)

        fh = logging.FileHandler(filename=f"{log_file_path}/server.log", mode='w')
        fh.setFormatter(format)
        fh.setLevel(logging.INFO)

        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(format)
        sh.setLevel(logging.DEBUG)

        self.logger.addHandler(fh)
        self.logger.addHandler(sh)
        self.logger.info("Starting server logging")

