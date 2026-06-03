"""
Synthetic 10-class dataset for the federated learning demo.

Design:
  - Each class = a Gaussian cluster with a distinct one-hot-style mean
  - Signal strength / noise tuned so IID converges fast but non-IID
    starts ~20% lower and catches up over rounds — the core FL story
  - Fully offline, no downloads
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

N_CLASSES = 10
FEAT_DIM  = 64
SIGNAL    = 2.5   # class mean magnitude
NOISE     = 1.5   # Gaussian noise std — tuned to show IID/non-IID gap


def _make_means():
    means = np.zeros((N_CLASSES, FEAT_DIM), np.float32)
    block = FEAT_DIM // N_CLASSES
    for i in range(N_CLASSES):
        means[i, i * block:(i + 1) * block] = SIGNAL
    return means


_MEANS = _make_means()


def _build(n: int, seed: int = 0):
    rng    = np.random.default_rng(seed)
    labels = rng.integers(0, N_CLASSES, n).astype(np.int64)
    noise  = rng.standard_normal((n, FEAT_DIM)).astype(np.float32) * NOISE
    X      = _MEANS[labels] + noise
    return torch.from_numpy(X), torch.from_numpy(labels)


_TRAIN_X, _TRAIN_Y = _build(8000, seed=1)
_TEST_X,  _TEST_Y  = _build(1500, seed=99)
_TESTLOADER = DataLoader(TensorDataset(_TEST_X, _TEST_Y), batch_size=256)


def load_partition(client_id: int, num_clients: int = 5, non_iid: bool = True):
    labels = _TRAIN_Y.numpy()

    if non_iid:
        dominant  = [(client_id * 2) % N_CLASSES, (client_id * 2 + 1) % N_CLASSES]
        dom_idx   = np.where(np.isin(labels, dominant))[0]
        other_idx = np.where(~np.isin(labels, dominant))[0]
        rng    = np.random.default_rng(client_id * 31)
        n_dom  = min(len(dom_idx), 400)
        n_other = n_dom // 5
        chosen = np.concatenate([
            rng.choice(dom_idx,   n_dom,   replace=False),
            rng.choice(other_idx, n_other, replace=False),
        ])
        rng.shuffle(chosen)
    else:
        per_c  = len(labels) // num_clients
        chosen = np.arange(client_id * per_c, (client_id + 1) * per_c)

    ds = TensorDataset(_TRAIN_X[chosen], _TRAIN_Y[chosen])
    return DataLoader(ds, batch_size=64, shuffle=True), _TESTLOADER


def describe_partition(client_id: int, num_clients: int = 5, non_iid: bool = True):
    tl, _ = load_partition(client_id, num_clients, non_iid)
    counts = [0] * N_CLASSES
    for _, ys in tl:
        for l in ys: counts[l.item()] += 1
    total    = sum(counts)
    dominant = [(client_id * 2) % N_CLASSES, (client_id * 2 + 1) % N_CLASSES]
    top      = sorted(range(N_CLASSES), key=lambda i: -counts[i])[:3]
    info     = "  ".join(f"cls{i}:{counts[i]}" for i in top)
    print(f"  Client {client_id}  n={total:4d}  dominant={dominant}  top -> {info}")
