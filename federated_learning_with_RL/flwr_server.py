from typing import List, Optional, Dict, Tuple

import flwr as fl


class FedAvgWithRL(fl.server.strategy.FedAvg):
    """
    """

    def __init__(self, rl_agent=None, fraction_fit: float = 1.0, fraction_evaluate: float = 1.0, min_fit_clients: int = 2, min_evaluate_clients: int = 2, min_available_clients: int = 2, local_epochs: int = 1):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            on_fit_config_fn=self._on_fit_config,
            evaluate_metrics_aggregation_fn=self.aggregate_eval_metrics,
            fit_metrics_aggregation_fn=self.aggregate_fit_metrics,
        )
        # RL-related state
        self.rl_agent = rl_agent
        self.default_local_epochs = local_epochs
        self.last_global_accuracy = 0.0
        self.actions_log = []  # store per-round selected hyperparameters

    def configure_fit(self, server_round: int, parameters, client_manager):
        # Get default instructions
        ins = super().configure_fit(server_round, parameters, client_manager)
        # Optionally filter client sampling via RL (client selection)
        if self.rl_agent is not None and hasattr(self.rl_agent, "select"):
            client_ids = [cid for cid, _ in ins]
            selected = self.rl_select_clients(client_ids, server_round)
            if selected:
                ins = [(cid, cfg) for cid, cfg in ins if cid in selected]
        return ins

    def rl_select_clients(self, client_ids: List[str], server_round: int) -> Optional[List[str]]:
        # Placeholder RL policy: selects all clients (no filtering)
        # Replace with calls to `self.rl_agent.select(client_ids, state)` when ready.
        return client_ids

    # DQN: provide fit config per round based on last_global_accuracy
    def _on_fit_config(self, rnd: int) -> Dict[str, str]:
        cfg = {"local_epochs": str(self.default_local_epochs)}
        # If rl_agent provides hyperparameter action selection
        if self.rl_agent is not None and hasattr(self.rl_agent, "select_action"):
            lr, epochs, frac, a_idx = self.rl_agent.select_action(self.last_global_accuracy)
            # Adjust fraction_fit dynamically (bounded by [0.1,1.0])
            self.fraction_fit = max(0.1, min(1.0, frac))
            # Provide learning rate and epochs to clients via config
            cfg.update({
                "learning_rate": str(lr),
                "local_epochs": str(epochs),
                "action_index": str(a_idx),
            })
            # Log selected action for this round
            self.actions_log.append({
                "round": rnd,
                "learning_rate": lr,
                "local_epochs": epochs,
                "fraction_fit": float(self.fraction_fit),
                "action_index": a_idx,
                "state_last_accuracy": float(self.last_global_accuracy),
            })
        return cfg

    @staticmethod
    def aggregate_eval_metrics(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
        # metrics: list of (num_examples, {"accuracy": acc})
        total_examples = sum(num for num, _ in metrics)
        if total_examples == 0:
            return {}
        # Weighted average by number of examples
        acc_sum = sum(m.get("accuracy", 0.0) * num for num, m in metrics)
        acc = acc_sum / total_examples
        return {"accuracy": acc}

    @staticmethod
    def aggregate_fit_metrics(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
        # Provide a placeholder to avoid warnings; you can add custom metrics in client fit later
        return {}

    # Hook: after evaluate aggregation, update RL agent with reward
    def aggregate_evaluate(self, server_round: int, results, failures):
        agg = super().aggregate_evaluate(server_round, results, failures)
        # agg is Optional[EvaluateRes]; access aggregated metrics through strategy history automatically.
        # But we can update last_global_accuracy here if available in results.
        try:
            # When evaluate_metrics_aggregation_fn is used, strategy history is updated.
            # We recompute here from results to update RL.
            metrics: List[Tuple[int, Dict[str, float]]] = []
            for res in results:
                num = res.num_examples
                m = res.metrics
                metrics.append((num, m))
            aggregated = self.aggregate_eval_metrics(metrics)
            acc = aggregated.get("accuracy", 0.0)
            
            # Log epsilon BEFORE calling observe (which updates epsilon)
            epsilon_before = getattr(self.rl_agent, "epsilon", 0.0) if self.rl_agent else 0.0
            
            # Update RL agent with reward
            if self.rl_agent is not None and hasattr(self.rl_agent, "observe"):
                self.rl_agent.observe(acc)
            
            # Update state for next round
            self.last_global_accuracy = acc
            
            # Append reward and epsilon to latest action log entry
            if self.actions_log and len(self.actions_log) >= server_round:
                idx = server_round - 1  # actions_log is 0-indexed, rounds are 1-indexed
                self.actions_log[idx]["reward_accuracy"] = float(acc)
                self.actions_log[idx]["epsilon"] = float(epsilon_before)
        except Exception as e:
            print(f"Warning: Could not update RL metrics in aggregate_evaluate: {e}")
        return agg


def start_server(num_rounds: int = 3, local_epochs: int = 1, server_address: str = "0.0.0.0:8080"):
    strategy = FedAvgWithRL(local_epochs=local_epochs)
    return fl.server.start_server(server_address=server_address, strategy=strategy, config=fl.server.ServerConfig(num_rounds=num_rounds))
