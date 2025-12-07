from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import flwr as fl
import multiprocessing as mp

from .model import get_model


def resolve_device():
    # Use CUDA only when multiprocessing start method is 'spawn' to avoid re-init issues
    use_cuda = torch.cuda.is_available() and mp.get_start_method() == 'spawn'
    return torch.device("cuda" if use_cuda else "cpu")

DEVICE = resolve_device()


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, train_loader: DataLoader, val_loader: DataLoader):
        self.model = get_model().to(DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def get_parameters(self, config: Dict[str, str]):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for (k, v), p in zip(state_dict.items(), parameters):
            state_dict[k] = torch.tensor(p)
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config: Dict[str, str]):
        self.set_parameters(parameters)
        epochs = int(config.get("local_epochs", 1))
        self.model.train()
        for _ in range(epochs):
            for images, labels in self.train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
        return self.get_parameters({}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config: Dict[str, str]):
        self.set_parameters(parameters)
        self.model.eval()
        correct = 0
        total = 0
        loss_total = 0.0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss_total += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total if total > 0 else 0.0
        loss_avg = loss_total / total if total > 0 else 0.0
        return float(loss_avg), len(self.val_loader.dataset), {"accuracy": float(accuracy)}


def start_client(train_loader: DataLoader, val_loader: DataLoader, server_address: str = "0.0.0.0:8080"):
    client = FlowerClient(train_loader, val_loader)
    fl.client.start_numpy_client(server_address=server_address, client=client)
