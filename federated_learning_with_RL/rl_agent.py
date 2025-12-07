from typing import List
import random


class RandomRLAgent:
    """A minimal agent that selects a random subset of clients.

    This is a stub you can later replace with a DQN agent.
    """

    def __init__(self, select_fraction: float = 1.0, seed: int = 42):
        self.select_fraction = select_fraction
        self.rng = random.Random(seed)

    def select(self, client_ids: List[str], round_idx: int) -> List[str]:
        k = max(1, int(self.select_fraction * len(client_ids)))
        return self.rng.sample(client_ids, k)
