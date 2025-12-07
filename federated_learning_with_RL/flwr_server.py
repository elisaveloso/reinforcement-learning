from typing import List, Optional, Dict, Tuple

import flwr as fl


class FedAvgWithRL(fl.server.strategy.FedAvg):
    """Baseline FedAvg strategy with optional RL-driven client selection hook.

    To plug an RL agent, override `rl_select_clients` to return a subset of client IDs.
    """

    def __init__(self, rl_agent=None, fraction_fit: float = 1.0, fraction_evaluate: float = 1.0, min_fit_clients: int = 2, min_evaluate_clients: int = 2, min_available_clients: int = 2, local_epochs: int = 1):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            on_fit_config_fn=lambda rnd: {"local_epochs": str(local_epochs)},
            evaluate_metrics_aggregation_fn=self.aggregate_eval_metrics,
            fit_metrics_aggregation_fn=self.aggregate_fit_metrics,
        )
        self.rl_agent = rl_agent

    def configure_fit(self, server_round: int, parameters, client_manager):
        # Get default instructions
        ins = super().configure_fit(server_round, parameters, client_manager)
        # Optionally filter client sampling via RL
        if self.rl_agent is not None:
            # Collect client IDs
            client_ids = [cid for cid, _ in ins]
            selected = self.rl_select_clients(client_ids, server_round)
            if selected:
                ins = [(cid, cfg) for cid, cfg in ins if cid in selected]
        return ins

    def rl_select_clients(self, client_ids: List[str], server_round: int) -> Optional[List[str]]:
        # Placeholder RL policy: selects all clients (no filtering)
        # Replace with calls to `self.rl_agent.select(client_ids, state)` when ready.
        return client_ids

    @staticmethod
    def aggregate_eval_metrics(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
        # metrics: list of (num_examples, {"accuracy": acc})
        total_examples = sum(num for num, _ in metrics)
        if total_examples == 0:
            return {}
        # Weighted average by number of examples
        acc_sum = sum(m.get("accuracy", 0.0) * num for num, m in metrics)
        return {"accuracy": acc_sum / total_examples}

    @staticmethod
    def aggregate_fit_metrics(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
        # Provide a placeholder to avoid warnings; you can add custom metrics in client fit later
        return {}


def start_server(num_rounds: int = 3, local_epochs: int = 1, server_address: str = "0.0.0.0:8080"):
    strategy = FedAvgWithRL(local_epochs=local_epochs)
    return fl.server.start_server(server_address=server_address, strategy=strategy, config=fl.server.ServerConfig(num_rounds=num_rounds))
