import argparse
import multiprocessing as mp
import os
import time

from .data_loader import load_dataset, partition_dataset, get_loaders_for_subset
from .flwr_server import start_server, FedAvgWithRL
from .flwr_client import start_client
from .rl_agent import RandomRLAgent


DEFAULT_SERVER_ADDRESS = "0.0.0.0:8080"


def run_server(num_rounds: int, local_epochs: int, use_rl: bool, run_dir: str):
    rl_agent = RandomRLAgent(select_fraction=1.0) if use_rl else None
    strategy = FedAvgWithRL(rl_agent=rl_agent, local_epochs=local_epochs)
    from flwr import server
    history = server.start_server(server_address=DEFAULT_SERVER_ADDRESS, strategy=strategy, config=server.ServerConfig(num_rounds=num_rounds))
    # Save history to disk when server exits
    try:
        import json
        os.makedirs(run_dir, exist_ok=True)
        out_path = os.path.join(run_dir, "history.json")
        with open(out_path, "w") as f:
            json.dump({
                "losses_distributed": history.losses_distributed,
                "losses_centralized": history.losses_centralized,
                "metrics_distributed": history.metrics_distributed,
                "metrics_centralized": history.metrics_centralized,
            }, f, indent=2)
    except Exception:
        pass


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


if __name__ == "__main__":
    main()
