import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import ndarrays_to_parameters, Metrics
from model import MNISTNet, get_parameters
from typing import List, Tuple, Dict, Optional

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Weighted average of accuracy across all clients."""
    total     = sum(n for n, _ in metrics)
    accuracy  = sum(n * m["accuracy"] for n, m in metrics) / total
    return {"accuracy": accuracy}

def build_strategy(num_clients: int, num_rounds: int) -> FedAvg:
    initial_params = ndarrays_to_parameters(get_parameters(MNISTNet()))

    return FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        initial_parameters=initial_params,
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=lambda rnd: {
            "local_epochs": 3,
            "round": rnd,
        },
    )
