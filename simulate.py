"""
Federated Learning Demo — PyTorch + FedAvg / FedProx
=====================================================
Implements FL from scratch:
  - 5 simulated clients with private local data shards
  - FedAvg server-side aggregation (McMahan et al. 2017)
  - FedProx with proximal term µ/2 * ||w - w_global||²  (Li et al. 2020)
  - IID vs non-IID vs non-IID+FedProx convergence comparison
  - Saves fl_results.png
"""
import warnings; warnings.filterwarnings("ignore")
import copy, time, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch, torch.nn as nn

from model import MNISTNet, get_parameters, set_parameters
from data  import load_partition, describe_partition, _TESTLOADER

DEVICE       = torch.device("cpu")
NUM_CLIENTS  = 5
NUM_ROUNDS   = 12
LOCAL_EPOCHS = 2
LR           = 1e-2
MU           = 0.1   # FedProx proximal term coefficient


def train_client(global_model, trainloader):
    """Train a local copy of the global model and return its weights."""
    model = copy.deepcopy(global_model)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    crit  = nn.CrossEntropyLoss()
    model.train()
    for _ in range(LOCAL_EPOCHS):
        for X, y in trainloader:
            opt.zero_grad()
            crit(model(X), y).backward()
            opt.step()
    return get_parameters(model), len(trainloader.dataset)


def train_client_fedprox(global_model, trainloader, mu):
    """FedProx local step: adds µ/2 * ||w - w_global||² to the loss."""
    model         = copy.deepcopy(global_model)
    global_params = [p.detach().clone() for p in global_model.parameters()]
    opt           = torch.optim.Adam(model.parameters(), lr=LR)
    crit          = nn.CrossEntropyLoss()
    model.train()
    for _ in range(LOCAL_EPOCHS):
        for X, y in trainloader:
            opt.zero_grad()
            loss = crit(model(X), y)
            prox = sum(
                ((p - p0) ** 2).sum()
                for p, p0 in zip(model.parameters(), global_params)
            )
            (loss + (mu / 2) * prox).backward()
            opt.step()
    return get_parameters(model), len(trainloader.dataset)


def fedavg(all_params, all_sizes):
    """Weighted average of client weight arrays."""
    total = sum(all_sizes)
    return [
        sum(w * (n / total) for w, n in zip(layer, all_sizes))
        for layer in zip(*all_params)
    ]


def evaluate(model):
    model.eval()
    correct = 0
    with torch.no_grad():
        for X, y in _TESTLOADER:
            correct += (model(X).argmax(1) == y).sum().item()
    return correct / len(_TESTLOADER.dataset)


def run_experiment(non_iid: bool, label: str, mu: float = 0.0):
    algo = f"FedProx (µ={mu})" if mu > 0.0 else "FedAvg"
    print(f"\n{'='*62}")
    print(f"  {label}")
    print(f"{'='*62}")
    print(f"  Clients={NUM_CLIENTS}  Rounds={NUM_ROUNDS}  LocalEpochs={LOCAL_EPOCHS}  Algorithm={algo}")

    partitions = [load_partition(cid, NUM_CLIENTS, non_iid) for cid in range(NUM_CLIENTS)]

    if non_iid:
        print("\n  Data distribution (non-IID - each client dominated by 2 classes):")
        for cid in range(NUM_CLIENTS):
            describe_partition(cid, NUM_CLIENTS, non_iid=True)

    global_model = MNISTNet()
    accs = []

    for rnd in range(1, NUM_ROUNDS + 1):
        t0 = time.time()
        all_params, all_sizes = [], []
        for cid in range(NUM_CLIENTS):
            tl, _ = partitions[cid]
            if mu > 0.0:
                p, n = train_client_fedprox(global_model, tl, mu)
            else:
                p, n = train_client(global_model, tl)
            all_params.append(p)
            all_sizes.append(n)

        set_parameters(global_model, fedavg(all_params, all_sizes))
        acc = evaluate(global_model)
        accs.append(acc)

        bar = "#" * int(acc * 40)
        print(f"  Round {rnd:2d}  {bar:<40}  {acc*100:5.1f}%  ({time.time()-t0:.1f}s)")

    print(f"\n  Final accuracy: {accs[-1]*100:.1f}%")
    return list(range(1, NUM_ROUNDS + 1)), accs


def plot_results(iid_r, iid_a, niid_r, niid_a, fp_r, fp_a):
    purple, teal, orange = "#534AB7", "#1D9E75", "#E07B28"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Federated Learning Demo — FedAvg vs FedProx · {NUM_CLIENTS} clients · {NUM_ROUNDS} rounds",
        fontsize=14, fontweight="bold"
    )

    iid_pct  = [a * 100 for a in iid_a]
    niid_pct = [a * 100 for a in niid_a]
    fp_pct   = [a * 100 for a in fp_a]

    ax1.plot(iid_r,  iid_pct,  "o-",  color=purple, lw=2.5, ms=7, label="IID · FedAvg",         zorder=3)
    ax1.plot(niid_r, niid_pct, "s--", color=teal,   lw=2.5, ms=7, label="non-IID · FedAvg",     zorder=3)
    ax1.plot(fp_r,   fp_pct,   "^-",  color=orange, lw=2.5, ms=7, label=f"non-IID · FedProx µ={MU}", zorder=3)
    ax1.fill_between(fp_r, niid_pct, fp_pct, alpha=0.10, color=orange)
    ax1.set_xlabel("Communication round", fontsize=12)
    ax1.set_ylabel("Global accuracy (%)", fontsize=12)
    ax1.set_title("Accuracy over communication rounds", fontsize=12)
    ax1.set_ylim(0, 105)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_facecolor("#f9f9fb")
    ax1.annotate(f"{iid_pct[-1]:.1f}%",  (iid_r[-1],  iid_pct[-1]),
                 xytext=(-28, 6),   textcoords="offset points", color=purple, fontweight="bold", fontsize=10)
    ax1.annotate(f"{niid_pct[-1]:.1f}%", (niid_r[-1], niid_pct[-1]),
                 xytext=(-28, -14), textcoords="offset points", color=teal,   fontweight="bold", fontsize=10)
    ax1.annotate(f"{fp_pct[-1]:.1f}%",   (fp_r[-1],   fp_pct[-1]),
                 xytext=(4, 0),     textcoords="offset points", color=orange, fontweight="bold", fontsize=10)

    finals = [iid_pct[-1], niid_pct[-1], fp_pct[-1]]
    colors = [purple, teal, orange]
    labels = ["IID\n(FedAvg)", "non-IID\n(FedAvg)", f"non-IID\n(FedProx µ={MU})"]
    bars = ax2.bar(labels, finals, color=colors, width=0.45, edgecolor="white", lw=1.5)
    for bar, val in zip(bars, finals):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.8,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=13, fontweight="bold")

    fp_gain = fp_pct[-1] - niid_pct[-1]
    ax2.set_title(f"Final accuracy · FedProx gain over non-IID FedAvg: {fp_gain:+.1f}%", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.set_ylim(0, 110)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_facecolor("#f9f9fb")

    plt.tight_layout()
    out = "fl_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    try:
        os.makedirs("/mnt/user-data/outputs", exist_ok=True)
        plt.savefig("/mnt/user-data/outputs/fl_results.png", dpi=150, bbox_inches="tight")
    except OSError:
        pass
    plt.close()
    print(f"\n  Chart saved -> {out}")
    return out


if __name__ == "__main__":
    print("\n FEDERATED LEARNING DEMO")
    print(" PyTorch - FedAvg / FedProx from scratch - 10-class synthetic dataset\n")

    iid_r,  iid_a  = run_experiment(non_iid=False, label="Experiment 1 - IID  (uniform split)")
    niid_r, niid_a = run_experiment(non_iid=True,  label="Experiment 2 - non-IID (skewed split)")
    fp_r,   fp_a   = run_experiment(non_iid=True,  label=f"Experiment 3 - non-IID + FedProx (µ={MU})", mu=MU)

    print("\n" + "=" * 62)
    print("  RESULTS SUMMARY")
    print("=" * 62)
    print(f"  IID    final accuracy:              {iid_a[-1]*100:.1f}%")
    print(f"  non-IID final accuracy (FedAvg):    {niid_a[-1]*100:.1f}%")
    print(f"  non-IID final accuracy (FedProx):   {fp_a[-1]*100:.1f}%")
    gap    = (iid_a[-1]  - niid_a[-1]) * 100
    fp_gain = (fp_a[-1]  - niid_a[-1]) * 100
    print(f"  IID vs non-IID gap:                 {gap:+.1f}%")
    print(f"  FedProx gain over FedAvg (non-IID): {fp_gain:+.1f}%")
    print()
    print("  Core FL insights reproduced:")
    print("  -> non-IID data causes slower convergence in early rounds")
    print("  -> FedProx proximal term regularises local updates towards global model")
    print(f"  -> µ={MU} proximal coefficient — increase for stronger regularisation")

    plot_results(iid_r, iid_a, niid_r, niid_a, fp_r, fp_a)
    print("\nDone! Check fl_results.png for the comparison chart.")
