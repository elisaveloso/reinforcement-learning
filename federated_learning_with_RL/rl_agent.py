from typing import List, Tuple, Dict
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)


class DQNAgent:
    """DQN agent to choose FL hyperparameters each round.

    State: [last_global_accuracy] (scalar)
    Actions: discrete combinations of (learning_rate, local_epochs, fraction_fit)
    Reward: current global accuracy
    """

    def __init__(self, actions: List[Tuple[float, int, float]], seed: int = 42, gamma: float = 0.95, lr: float = 1e-3, epsilon_start: float = 1.0, epsilon_end: float = 0.05, epsilon_decay: float = 0.95):
        self.rng = random.Random(seed)
        self.actions = actions
        self.state_dim = 1
        self.action_dim = len(actions)
        self.q = QNetwork(self.state_dim, self.action_dim)
        self.opt = optim.Adam(self.q.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.prev_state = None
        self.prev_action = None

    def select_action(self, last_accuracy: float) -> Tuple[float, int, float, int]:
        state = torch.tensor([last_accuracy], dtype=torch.float32)
        if self.rng.random() < self.epsilon:
            a_idx = self.rng.randrange(self.action_dim)
        else:
            with torch.no_grad():
                qvals = self.q(state)
                a_idx = int(torch.argmax(qvals).item())
        self.prev_state = state
        self.prev_action = a_idx
        lr, epochs, frac = self.actions[a_idx]
        return lr, epochs, frac, a_idx

    def observe(self, new_accuracy: float):
        """One-step TD update using reward=new_accuracy and next_state=[new_accuracy]."""
        if self.prev_state is None or self.prev_action is None:
            # First round: no previous state/action yet
            return
        reward = new_accuracy
        next_state = torch.tensor([new_accuracy], dtype=torch.float32)
        # Q-learning target: r + gamma * max_a' Q(next_state, a')
        with torch.no_grad():
            q_next = self.q(next_state)
            target = reward + self.gamma * torch.max(q_next)
        q_prev = self.q(self.prev_state)[self.prev_action]
        loss = (q_prev - target) ** 2
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        # decay epsilon after learning step
        old_epsilon = self.epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        print(f"  [DQN] Reward: {reward:.4f}, Loss: {loss.item():.4f}, Epsilon: {old_epsilon:.4f} -> {self.epsilon:.4f}")


class RandomRLAgent:
    """Fallback random selector for client sampling (unchanged)."""

    def __init__(self, select_fraction: float = 1.0, seed: int = 42):
        self.select_fraction = select_fraction
        self.rng = random.Random(seed)

    def select(self, client_ids: List[str], round_idx: int) -> List[str]:
        k = max(1, int(self.select_fraction * len(client_ids)))
        return self.rng.sample(client_ids, k)
