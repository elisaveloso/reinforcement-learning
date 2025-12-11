import argparse
import multiprocessing as mp
import os
import time
import json
import csv

from .data_loader import load_dataset, partition_dataset, get_loaders_for_subset
from .flwr_server import start_server, FedAvgWithRL
from .flwr_client import start_client
from .rl_agent import RandomRLAgent


DEFAULT_SERVER_ADDRESS = "0.0.0.0:8080"


def run_server(num_rounds: int, local_epochs: int, use_rl: bool, run_dir: str):
    # Initialize RL agent for hyperparameter selection if enabled
    if use_rl:
        from .rl_agent import DQNAgent
        # Define a small discrete action grid: (learning_rate, local_epochs, fraction_fit)
        actions = [
            (0.001, 1, 1.0), (0.001, 2, 0.75), (0.001, 1, 0.5),
            (0.0005, 1, 1.0), (0.0005, 2, 0.75), (0.0005, 1, 0.5),
            (0.0001, 1, 1.0), (0.0001, 2, 0.75), (0.0001, 1, 0.5),
            (0.01, 1, 1.0), (0.01, 2, 0.75), (0.01, 1, 0.5),
            (0.1, 1, 1.0), (0.1, 2, 0.75), (0.1, 1, 0.5),
            (0.2, 1, 1.0), (0.2, 2, 0.75), (0.2, 1, 0.5)
        ]
        rl_agent = DQNAgent(actions=actions)
    else:
        rl_agent = None
    strategy = FedAvgWithRL(rl_agent=rl_agent, local_epochs=local_epochs)
    from flwr import server
    history = server.start_server(server_address=DEFAULT_SERVER_ADDRESS, strategy=strategy, config=server.ServerConfig(num_rounds=num_rounds)) #ao  inves de pegar FedAvg, pega o agente RL
    
    try:
        import json, csv
        os.makedirs(run_dir, exist_ok=True)
        print(f"[Server] Saving artifacts to {run_dir}")
        
        # Save history if available
        if history is not None:
            out_path = os.path.join(run_dir, "history.json")
            with open(out_path, "w") as f:
                json.dump({
                    "losses_distributed": history.losses_distributed,
                    "losses_centralized": history.losses_centralized,
                    "metrics_distributed": history.metrics_distributed,
                    "metrics_centralized": history.metrics_centralized,
                }, f, indent=2)
            print(f"[Server] Saved history.json")
        
        # Save RL actions and CSV if RL is enabled
        if use_rl and hasattr(strategy, "actions_log") and strategy.actions_log:
            # Save JSON
            with open(os.path.join(run_dir, "actions.json"), "w") as fa:
                json.dump(strategy.actions_log, fa, indent=2)
            print(f"[Server] Saved actions.json with {len(strategy.actions_log)} rounds")
            
            # Save CSV
            csv_path = os.path.join(run_dir, "rl_training.csv")
            header = [
                "round",
                "state_last_accuracy",
                "learning_rate",
                "local_epochs",
                "fraction_fit",
                "action_index",
                "reward_accuracy",
                "epsilon",
            ]
            with open(csv_path, "w", newline="") as cf:
                writer = csv.DictWriter(cf, fieldnames=header)
                writer.writeheader()
                for row in strategy.actions_log:
                    writer.writerow({
                        "round": row.get("round"),
                        "state_last_accuracy": row.get("state_last_accuracy"),
                        "learning_rate": row.get("learning_rate"),
                        "local_epochs": row.get("local_epochs"),
                        "fraction_fit": row.get("fraction_fit"),
                        "action_index": row.get("action_index"),
                        "reward_accuracy": row.get("reward_accuracy"),
                        "epsilon": row.get("epsilon"),
                    })
            print(f"[Server] Saved rl_training.csv")
    except Exception as e:
        print(f"[Server] Error saving artifacts: {e}")


def run_client(subset):
    train_loader, val_loader = get_loaders_for_subset(subset)
    start_client(train_loader, val_loader, server_address=DEFAULT_SERVER_ADDRESS)


def main():
    # Use 'spawn' to allow CUDA initialization inside subprocesses if available
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # Start method may already be set; ignore
        pass
    parser = argparse.ArgumentParser(description="Federated Learning with RL - Binary Classification")
    parser.add_argument("--data-root", type=str, default=os.path.join(os.path.dirname(__file__), "data"), help="Path to dataset root containing class folders.")
    parser.add_argument("--num-clients", type=int, default=4, help="Number of simulated clients.")
    parser.add_argument("--num-rounds", type=int, default=3, help="Federated rounds.")
    parser.add_argument("--local-epochs", type=int, default=1, help="Local epochs per round.")
    parser.add_argument("--use-rl", action="store_true", help="Enable RL-driven client selection (stub).")
    parser.add_argument("--tag", type=str, default="", help="Optional tag/name for this run to help identify configurations.")
    args = parser.parse_args()

    dataset = load_dataset(args.data_root)
    subsets = partition_dataset(dataset, args.num_clients)

    # Prepare results run directory with timestamp and optional tag
    ts = time.strftime("%Y%m%d-%H%M%S")
    tag = f"-{args.tag}" if args.tag else ""
    run_dir = os.path.join(os.path.dirname(__file__), "results", f"run-{ts}{tag}")
    os.makedirs(run_dir, exist_ok=True)

    # Write a pointer to latest run for convenience
    try:
        results_root = os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(results_root, exist_ok=True)
        with open(os.path.join(results_root, "latest.txt"), "w") as f:
            f.write(run_dir)
        print(f"Results directory: {run_dir}")
    except Exception:
        pass

    # Save configuration used for this run
    try:
        import json
        cfg = {
            "data_root": args.data_root,
            "num_clients": args.num_clients,
            "num_rounds": args.num_rounds,
            "local_epochs": args.local_epochs,
            "use_rl": args.use_rl,
            "server_address": DEFAULT_SERVER_ADDRESS,
            "timestamp": ts,
            "tag": args.tag,
        }
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)
    except Exception:
        pass

    # Start server
    server_proc = mp.Process(target=run_server, args=(args.num_rounds, args.local_epochs, args.use_rl, run_dir))
    server_proc.start()

    # Start clients
    client_procs = []
    for subset in subsets:
        p = mp.Process(target=run_client, args=(subset,))
        p.start()
        client_procs.append(p)

    # Wait for clients
    for p in client_procs:
        p.join()

    # Stop server
    server_proc.join()

    # After server finishes, append a summary entry to a global index for easy comparison across runs
    try:
        
        # Load history
        hist_path = os.path.join(run_dir, "history.json")
        cfg_path = os.path.join(run_dir, "config.json")
        if os.path.isfile(hist_path) and os.path.isfile(cfg_path):
            with open(hist_path, "r") as f:
                hist = json.load(f)
            with open(cfg_path, "r") as f:
                cfg = json.load(f)

            # Extract metrics
            losses = hist.get("losses_distributed", [])
            final_loss = losses[-1][1] if losses else None
            acc_series = hist.get("metrics_distributed", {}).get("accuracy", [])
            final_acc = acc_series[-1][1] if acc_series else None
            avg_acc = sum(v for _, v in acc_series) / len(acc_series) if acc_series else None
            best_acc = max((v for _, v in acc_series), default=None)

            # Global index path
            results_root = os.path.join(os.path.dirname(__file__), "results")
            index_csv = os.path.join(results_root, "index.csv")
            # Write header if file does not exist
            header = [
                "timestamp",
                "tag",
                "run_dir",
                "data_root",
                "num_clients",
                "num_rounds",
                "local_epochs",
                "use_rl",
                "final_loss",
                "final_acc",
                "avg_acc",
                "best_acc",
            ]
            write_header = not os.path.isfile(index_csv)
            os.makedirs(results_root, exist_ok=True)
            with open(index_csv, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(header)
                writer.writerow([
                    cfg.get("timestamp"),
                    cfg.get("tag"),
                    run_dir,
                    cfg.get("data_root"),
                    cfg.get("num_clients"),
                    cfg.get("num_rounds"),
                    cfg.get("local_epochs"),
                    cfg.get("use_rl"),
                    final_loss,
                    final_acc,
                    avg_acc,
                    best_acc,
                ])
            print(f"Appended summary to {index_csv}")
    except Exception as e:
        # Keep training robust; just print a short message
        print(f"Could not append run summary: {e}")


if __name__ == "__main__":
    main()
