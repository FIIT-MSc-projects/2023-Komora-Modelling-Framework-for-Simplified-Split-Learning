import flwr as fl
from flwr.server.server import Server, fit_clients
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.strategy import Strategy, FedAvg
from flwr.server.client_manager import SimpleClientManager
from flwr.common import FitIns, EvaluateIns, ndarrays_to_parameters, parameters_to_ndarrays, Parameters
import time
from stage import Stage

class CustomStrategy(Strategy):

    def initialize_parameters(self, client_manager):
        print("initialize_parameters")
        return None

    def configure_fit(self, server_round, parameters, client_manager):
        fit_ins = FitIns(parameters, {})

        # Sample clients
        sample_size, min_num_clients = 2,2
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(self, server_round, results, failures):
        print("aggregate_fit")
        print(type(results))
       
        print("GONNA SLEEP")
        time.sleep(10)
        print("Now it will fail")
        return None

    def configure_evaluate(self, server_round, parameters, client_manager):
        print("configure_evaluate\n\n\n\n\n")
        print(type(parameters))
        return None

    def aggregate_evaluate(self, server_round, results, failures):
        print("aggregate_evaluate")
        return None

    def evaluate(self, server_round, parameters):
        print("evaluate")
        return None


class CustomServer(Server):
    def __init__(self, client_manager, strategy):
        self._client_manager = client_manager
        self.parameters = Parameters(
            tensors=[], tensor_type="numpy.ndarray"
        )
        self.strategy = strategy
        self.max_workers = None
        self.stage = Stage.RECORD_ALIGNMENT
 
server = CustomServer(client_manager=SimpleClientManager(), strategy=CustomStrategy())

fl.server.start_server(config=fl.server.ServerConfig(num_rounds=50), server_address='127.0.0.1:8081', strategy=CustomStrategy()
                    #    , server=server
                       )

