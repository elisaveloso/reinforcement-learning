# Basic Federated Learning Project with Reinforcement Learning (DQN)

This project implements a Federated Learning system where a server uses a Reinforcement Learning agent (DQN) to dynamically select the number of participating clients in each training round.

## Structure

- `dataset.py`: Loads and partitions the image dataset (weed vs non-weed).
- `model.py`: Defines the Convolutional Neural Network (CNN) architecture.
- `client.py`: Defines the Flower Client that trains the model locally.
- `rl_agent.py`: Implements the DQN (Deep Q-Network) agent.
- `main.py`: Configures the simulation, the custom strategy, and starts training.

## How to Run

1. Make sure you are in the correct virtual environment:
    ```bash
    source ../venv/bin/activate
    ```

2. Install dependencies (if not already installed):
    ```bash
    pip install flwr torch torchvision numpy
    ```

3. Run the simulation:
    ```bash
    python main.py
    ```

## How It Works

- The project uses a custom Federated Learning loop (not using `flwr.simulation` to avoid memory overhead in constrained environments).
- The **RL Agent** observes the current state (round number and previous round accuracy).
- It chooses an **Action**: how many clients (from 1 to 5) should participate in this round.
- The **Main Loop** randomly selects clients and coordinates training:
     - Each client trains locally on its data partition.
     - Parameters are aggregated by the server (FedAvg).
     - Global accuracy is estimated by averaging the clients' accuracies.
- The agent receives a **Reward** based on the achieved accuracy minus a cost per client used.
