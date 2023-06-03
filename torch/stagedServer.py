import flwr as fl
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.strategy import Strategy, FedAvg
from flwr.common import FitIns, EvaluateIns, ndarrays_to_parameters, parameters_to_ndarrays
from stage import Stage

class CustomStrategy(Strategy):

    def __init__(self, initial_parameters=None):
        super().__init__()
        self.initial_parameters = initial_parameters
        self.stage = Stage.FORWARD

    def initialize_parameters(self, client_manager):
        print("initialize_parameters")
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters

    def configure_fit(self, server_round, parameters, client_manager):
        print("configure_fit")
        fit_ins = FitIns(parameters, {"stage": self.stage.value})

        if self.stage == Stage.FORWARD:
            self.stage = Stage.BACKWARD
        elif self.stage == Stage.BACKWARD:
            self.stage = Stage.FORWARD

        clients = client_manager.sample(num_clients=2, min_num_clients=2)
        return [(client, fit_ins) for client in clients] # Return client/config pairs

    def aggregate_fit(self, server_round, results, failures):
        print("aggregate_fit")

        print(len(results))

        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]


        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        return parameters_aggregated, metrics_aggregated

    def configure_evaluate(self, server_round, parameters, client_manager):
        print("configure_evaluate")
        evaluate_ins = EvaluateIns(parameters, {})

        # Sample clients
        clients = client_manager.sample(
            num_clients=2, min_num_clients=2
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(self, server_round, results, failures):
        print("aggregate_evaluate")
        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        return loss_aggregated, metrics_aggregated

    def evaluate(self, server_round, parameters):
        print("evaluate")
        return None

fl.server.start_server(config=fl.server.ServerConfig(num_rounds=10), server_address='127.0.0.1:3000', strategy=CustomStrategy())

