"""Compact MLP for the 64-dim synthetic feature dataset."""
import torch
import torch.nn as nn
from collections import OrderedDict

FEAT_DIM  = 64
N_CLASSES = 10


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEAT_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, N_CLASSES),
        )

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        if x.shape[1] != FEAT_DIM:
            x = x[:, :FEAT_DIM]
        return self.net(x)


MNISTNet = SimpleNet


def get_parameters(model):
    return [v.cpu().numpy() for _, v in model.state_dict().items()]


def set_parameters(model, parameters):
    sd = OrderedDict({
        k: torch.tensor(v)
        for k, v in zip(model.state_dict().keys(), parameters)
    })
    model.load_state_dict(sd, strict=True)
