import abc
import torch

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class ClientArguments():

    def __init__(
        self, 
        server_ref: torch.RRefType,
        server_model_refs: list,
        epochs: int
    ):
        self.server_ref = server_ref
        self.model_refs = server_model_refs
        self.epochs = epochs

    def get_server_ref(self):
        return self.server_ref
    
    def get_server_model_refs(self):
        return self.model_refs
    
    def get_epochs(self):
        return self.epochs
    
    def to_dict(self):
        return dotdict({
            "server_ref": self.server_ref,
            "model_refs": self.model_refs,
            "epochs": self.epochs
        })

class AbstractClient(abc.ABC):

    @abc.abstractmethod
    def __init__(self, args: ClientArguments):
        pass

    @abc.abstractmethod
    def train(self,last_alice_rref,last_alice_id):
        pass

    @abc.abstractmethod
    def give_weights(self):
        pass

    @abc.abstractmethod
    def eval(self):
        pass

    @abc.abstractmethod
    def load_data(self,args):
        pass

class StartClientArguments():
    def __init__(self, name, rank, world_size, port, address, input_model, output_model):
        self.name = name
        self.rank = rank
        self.world_size = world_size
        self.port = port
        self.address = address
        self.input_model = input_model
        self.output_model = output_model

    def get_name(self):
        return self.name
    
    def get_rank(self):
        return self.rank
    
    def get_world_size(self):
        return self.world_size
    
    def get_port(self):
        return self.port
    
    def get_address(self):
        return self.address
    
    def get_input_model(self):
        return self.input_model
    
    def get_output_model(self):
        return self.output_model


class ServerArguments():

    def __init__(
        self,
        client_num_in_total: int,
        client: AbstractClient,
        server_model: torch.nn.Module,
        epochs: int
    ):
        self.client_num_in_total = client_num_in_total
        self.client = client
        self.server_model = server_model
        self.epochs = epochs

    
    def get_client_num_in_total(self):
        return self.client_num_in_total
    
    def get_client(self):
        return self.client

    def get_server_model(self):
        return self.server_model
    
    def get_epochs(self):
        return self.epochs

class AbstractServer(abc.ABC):

    @abc.abstractmethod
    def __init__(self,args: ServerArguments):
        pass

    @abc.abstractmethod
    def train_request(self,client_id):
        pass

    @abc.abstractmethod
    def eval_request(self):
        pass

    @abc.abstractmethod
    def train(self,x):
        pass

class StartServerArguments():

    def __init__(
        self,
        port: str, 
        host: str,
        world_size: int,
        config: str,
        client: AbstractClient,
        server: AbstractServer,
        server_model: torch.nn.Module,
        epochs: int
    ):
        self.port = port
        self.host = host
        self.world_size = world_size
        self.config = config
        self.client = client
        self.server = server
        self.server_model = server_model
        self.epochs = epochs

    def get_port(self):
        return self.port

    def get_host(self):
        return self.host
    
    def get_world_size(self):
        return self.world_size
    
    def get_client_num_in_total(self):
        return self.world_size -1
    
    def get_config(self):
        return self.config
    
    def get_client(self):
        return self.client
    
    def get_server(self):
        return self.server
    
    def get_server_model(self):
        return self.server_model

    def get_epochs(self):
        return self.epochs