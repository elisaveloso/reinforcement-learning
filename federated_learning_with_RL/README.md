# Federated Learning + Reinforcement Learning (Binary Classification)

This is a minimal, runnable scaffold combining Flower (FL) and a stub RL agent for client selection. It trains a small CNN to classify images into two classes: `daninha` and `nao_daninha`.

## Project layout

- `model.py`: Simple PyTorch CNN for binary classification.
- `data_loader.py`: Loads `ImageFolder` from `data/` and partitions into client subsets.
- `flwr_client.py`: Flower client implementing `fit` and `evaluate`.
- `flwr_server.py`: Flower server strategy (`FedAvg` baseline) with an RL hook.
- `rl_agent.py`: Minimal random selector (placeholder for DQN).
- `train.py`: Orchestrates server and simulated clients on one machine.
- `data/`: Expected dataset with class subfolders `daninha/` and `nao_daninha/`.

## Requirements

Install from `requirements.txt` (you mentioned they're already installed):

```
flwr
stable-baselines3
gymnasium
numpy
pandas
matplotlib
opencv-python
scikit-learn
tqdm
seaborn
tensorboard
```

PyTorch should already be installed in your venv.

## Run

Activate your venv and start a small federated run (run from the repository root, not inside the `federated_learning_with_RL/` folder):

```
source venv/bin/activate
python -m federated_learning_with_RL.train --data-root federated_learning_with_RL/data --num-clients 4 --num-rounds 3 --local-epochs 1 --use-rl
```

- `--use-rl` enables the RL hook (currently a random selector that selects all clients if fraction=1.0).
- Adjust `--num-clients` to match how many partitions you want.

## Notes

- The CNN expects RGB images and resizes them to 240x320 (to match your images). Adjust `IMG_SIZE_H/IMG_SIZE_W` or transforms in `data_loader.py` if needed.
- `get_loaders_for_subset` splits each client subset into 80/20 train/val. You can change this.
- The RL piece is a placeholder; you can implement a DQN policy in `rl_agent.py` and call it from `FedAvgWithRL.configure_fit`.

## Next steps

- Replace `RandomRLAgent` with a proper DQN agent using `stable-baselines3` (or a custom PyTorch DQN) that maximizes global accuracy.
- Log metrics to TensorBoard for better visibility.
- Add persistence: save/load model weights per round.
- Add unit tests for data partitioning and client evaluation.
