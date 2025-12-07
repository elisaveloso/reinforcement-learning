import json
import os
import argparse
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)


def load_history(path: str):
    with open(path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Plot FL metrics from a run directory")
    parser.add_argument("--run-dir", type=str, default=os.path.join(HERE, "results"), help="Path to a run directory containing history.json, or the base results directory")
    args = parser.parse_args()

    run_dir = args.run_dir
    history_path = os.path.join(run_dir, "history.json")

    # If run_dir does not directly contain history.json, try latest.txt or scan for latest run
    if not os.path.isfile(history_path):
        latest_txt = os.path.join(run_dir, "latest.txt")
        if os.path.isfile(latest_txt):
            with open(latest_txt, "r") as f:
                run_dir = f.read().strip()
                history_path = os.path.join(run_dir, "history.json")
        else:
            # Scan for latest run-* directory
            candidates = [d for d in os.listdir(run_dir) if d.startswith("run-") and os.path.isdir(os.path.join(run_dir, d))]
            if not candidates:
                raise FileNotFoundError(f"No run directories found under {run_dir}")
            candidates.sort(reverse=True)
            run_dir = os.path.join(run_dir, candidates[0])
            history_path = os.path.join(run_dir, "history.json")

    if not os.path.isfile(history_path):
        raise FileNotFoundError(f"history.json not found in {run_dir}")

    print(f"Using run directory: {run_dir}")
    hist = load_history(history_path)
    rounds = []
    losses = []
    accs = []

    # losses_distributed is list of tuples: (round, loss)
    for r, l in hist.get("losses_distributed", []):
        rounds.append(r)
        losses.append(l)

    # metrics_distributed is dict: {"accuracy": [(round, value), ...]}
    metrics = hist.get("metrics_distributed", {})
    acc_series = metrics.get("accuracy", [])
    accs = [v for _, v in acc_series]

    # Plot
    plt.figure(figsize=(8, 4))
    if losses:
        plt.plot(rounds, losses, label="Loss (distributed)")
    if accs:
        # If no losses were recorded, derive rounds from accuracy series
        if not rounds:
            rounds = list(range(1, len(accs) + 1))
        plt.plot(rounds[:len(accs)], accs, label="Accuracy (distributed)")
    plt.xlabel("Round")
    plt.legend()
    plt.title("Federated Training Metrics")
    plt.tight_layout()

    out_png = os.path.join(run_dir, "metrics.png")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png)
    print(f"Saved plot to {out_png}")


if __name__ == "__main__":
    main()
