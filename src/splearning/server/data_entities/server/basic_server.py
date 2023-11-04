import time
import torch
from torch.distributed.rpc import RRef
from splearning.utils.data_structures import AbstractServer
from splearning.utils.data_structures import AbstractServerStrategy
from splearning.utils.data_structures import ClientArguments
from splearning.utils.data_structures import ServerArguments

class BasicServer(AbstractServer):
    def __init__(self, args: ServerArguments):

        self.server_ref: RRef = RRef(self)
        self.model: torch.nn.Module = args.get_server_model()()
        self.total_client_num: int  = args.get_total_client_num()
        self.strategy: AbstractServerStrategy = args.get_server_strategy()()
        self.lock = False
        self.lock2 = False

        self.__init_clients(
            client_declaration=args.get_client_declaration(), 
            epochs=args.get_epochs(), 
            total_client_num=args.get_total_client_num(), 
            clients_configs=args.get_clients_configs()
        )

    def train_request(self, **kwargs):
        self.strategy.execute_train_request(self.clients, **kwargs)


    def eval_request(self, **kwargs):
        self.strategy.execute_eval_request(self.clients, self.total_client_num, **kwargs)

    def train(self,x):
        return self.model(x)  
    
    def __init_clients(self, client_declaration, epochs, total_client_num, clients_configs):
        server_model_refs = list(map(lambda x: RRef(x),self.model.parameters()))

        if clients_configs is None:
            client_args = ClientArguments(self.server_ref, server_model_refs, epochs),       
            self.clients = self.strategy.create_clients(client_declaration, client_args, total_client_num)

        else:
            self.clients = self.strategy.create_clients_from_configs(clients_configs, self.server_ref, server_model_refs)

        
