import flwr as fl
import torch
import torch.nn as nn
from model import MNISTNet, get_parameters, set_parameters
from data import load_partition
from typing import Dict, List, Tuple
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id: int, num_clients: int = 5, non_iid: bool = True):
        self.client_id   = client_id
        self.model       = MNISTNet().to(DEVICE)
        self.trainloader, self.testloader = load_partition(client_id, num_clients, non_iid)
        print(f"  Client {client_id}: {len(self.trainloader.dataset)} train samples | "
              f"dominant digits: {[(client_id*2)%10, (client_id*2+1)%10] if non_iid else 'all'}")

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        return get_parameters(self.model)

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        set_parameters(self.model, parameters)
        epochs    = int(config.get("local_epochs", 3))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        total_loss = 0.0
        batches    = 0
        for _ in range(epochs):
            for images, labels in self.trainloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(self.model(images), labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                batches    += 1

        avg_loss = total_loss / max(batches, 1)
        return get_parameters(self.model), len(self.trainloader.dataset), {"train_loss": avg_loss}

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        set_parameters(self.model, parameters)
        criterion = nn.CrossEntropyLoss()
        self.model.eval()
        loss, correct = 0.0, 0

        with torch.no_grad():
            for images, labels in self.testloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs  = self.model(images)
                loss    += criterion(outputs, labels).item()
                correct += (outputs.argmax(1) == labels).sum().item()

        accuracy = correct / len(self.testloader.dataset)
        return float(loss), len(self.testloader.dataset), {"accuracy": accuracy}
