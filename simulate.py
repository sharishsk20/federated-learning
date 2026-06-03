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

DEVICE         = torch.device("cpu")
NUM_CLIENTS    = 5
NUM_ROUNDS     = 12
LOCAL_EPOCHS   = 2
LR             = 1e-2
MU             = 0.1   # FedProx proximal term coefficient
ACC_THRESHOLD  = 70.0  # % — used in convergence-speed annotation


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


def _l2_drift(params_after, params_before):
    """L2 norm of weight delta between two parameter lists."""
    return float(np.sqrt(sum(((a - b) ** 2).sum()
                             for a, b in zip(params_after, params_before))))


def run_experiment(non_iid: bool, label: str, mu: float = 0.0):
    """Returns (rounds, accs, client_drifts) where client_drifts[round][client] = L2 drift."""
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

    global_model  = MNISTNet()
    accs          = []
    client_drifts = []   # [round_idx][client_idx]

    for rnd in range(1, NUM_ROUNDS + 1):
        t0               = time.time()
        global_params_np = get_parameters(global_model)   # snapshot before local steps
        all_params, all_sizes, round_drifts = [], [], []

        for cid in range(NUM_CLIENTS):
            tl, _ = partitions[cid]
            if mu > 0.0:
                p, n = train_client_fedprox(global_model, tl, mu)
            else:
                p, n = train_client(global_model, tl)
            round_drifts.append(_l2_drift(p, global_params_np))
            all_params.append(p)
            all_sizes.append(n)

        client_drifts.append(round_drifts)
        set_parameters(global_model, fedavg(all_params, all_sizes))
        acc = evaluate(global_model)
        accs.append(acc)

        mean_drift = np.mean(round_drifts)
        bar = "#" * int(acc * 40)
        print(f"  Round {rnd:2d}  {bar:<40}  {acc*100:5.1f}%  drift={mean_drift:.3f}  ({time.time()-t0:.1f}s)")

    print(f"\n  Final accuracy: {accs[-1]*100:.1f}%")
    return list(range(1, NUM_ROUNDS + 1)), accs, client_drifts


def _rounds_to(accs_pct, threshold=ACC_THRESHOLD):
    """First round index (1-based) where accuracy crosses threshold, else None."""
    for i, a in enumerate(accs_pct):
        if a >= threshold:
            return i + 1
    return None


def plot_results(iid_r, iid_a, iid_cd,
                 niid_r, niid_a, niid_cd,
                 fp_r,   fp_a,   fp_cd):
    purple, teal, orange = "#534AB7", "#1D9E75", "#E07B28"
    rounds = list(range(1, NUM_ROUNDS + 1))

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        f"Federated Learning — FedAvg vs FedProx · {NUM_CLIENTS} clients · {NUM_ROUNDS} rounds",
        fontsize=14, fontweight="bold", y=1.01
    )
    ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # ── Panel 1: accuracy curves ──────────────────────────────────────────
    iid_pct  = [a * 100 for a in iid_a]
    niid_pct = [a * 100 for a in niid_a]
    fp_pct   = [a * 100 for a in fp_a]

    ax1.plot(iid_r,  iid_pct,  "o-",  color=purple, lw=2.5, ms=7, label="IID · FedAvg",              zorder=3)
    ax1.plot(niid_r, niid_pct, "s--", color=teal,   lw=2.5, ms=7, label="non-IID · FedAvg",          zorder=3)
    ax1.plot(fp_r,   fp_pct,   "^-",  color=orange, lw=2.5, ms=7, label=f"non-IID · FedProx µ={MU}", zorder=3)
    ax1.fill_between(fp_r, niid_pct, fp_pct, alpha=0.10, color=orange)
    ax1.axhline(ACC_THRESHOLD, color="#aaa", lw=1, ls=":", zorder=1)
    ax1.text(NUM_ROUNDS * 0.02, ACC_THRESHOLD + 1, f"{ACC_THRESHOLD:.0f}% threshold",
             fontsize=8, color="#888")
    ax1.set_xlabel("Communication round", fontsize=11)
    ax1.set_ylabel("Global accuracy (%)", fontsize=11)
    ax1.set_title("Accuracy over rounds", fontsize=12)
    ax1.set_ylim(0, 105)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.set_facecolor("#f9f9fb")
    for pct, col, dy in [(iid_pct, purple, 6), (niid_pct, teal, -14), (fp_pct, orange, 6)]:
        ax1.annotate(f"{pct[-1]:.1f}%", (rounds[-1], pct[-1]),
                     xytext=(-28, dy), textcoords="offset points",
                     color=col, fontweight="bold", fontsize=9)

    # ── Panel 2: mean weight drift per round ─────────────────────────────
    mean_drift_niid = [np.mean(rd) for rd in niid_cd]
    mean_drift_fp   = [np.mean(rd) for rd in fp_cd]
    mean_drift_iid  = [np.mean(rd) for rd in iid_cd]

    ax2.plot(rounds, mean_drift_iid,  "o-",  color=purple, lw=2,   ms=6, label="IID · FedAvg",              zorder=3)
    ax2.plot(rounds, mean_drift_niid, "s--", color=teal,   lw=2,   ms=6, label="non-IID · FedAvg",          zorder=3)
    ax2.plot(rounds, mean_drift_fp,   "^-",  color=orange, lw=2,   ms=6, label=f"non-IID · FedProx µ={MU}", zorder=3)
    ax2.fill_between(rounds, mean_drift_fp, mean_drift_niid, alpha=0.10, color=orange)
    ax2.set_xlabel("Communication round", fontsize=11)
    ax2.set_ylabel("Mean L2 weight drift", fontsize=11)
    ax2.set_title("Client weight drift from global model\n(higher = more client heterogeneity)", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_facecolor("#f9f9fb")

    # ── Panels 3 & 4: per-client drift heatmaps ──────────────────────────
    niid_mat = np.array(niid_cd).T   # (NUM_CLIENTS, NUM_ROUNDS)
    fp_mat   = np.array(fp_cd).T

    vmax = max(niid_mat.max(), fp_mat.max())
    vmin = min(niid_mat.min(), fp_mat.min())

    for ax, mat, title, algo_col in [
        (ax3, niid_mat, "Client drift heatmap — non-IID · FedAvg",          teal),
        (ax4, fp_mat,   f"Client drift heatmap — non-IID · FedProx µ={MU}", orange),
    ]:
        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd",
                       vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_yticks(range(NUM_CLIENTS))
        ax.set_yticklabels([f"Client {i}" for i in range(NUM_CLIENTS)], fontsize=9)
        ax.set_xticks(range(NUM_ROUNDS))
        ax.set_xticklabels(rounds, fontsize=8)
        ax.set_xlabel("Communication round", fontsize=10)
        ax.set_title(title, fontsize=11, color=algo_col, fontweight="bold")
        for r in range(NUM_ROUNDS):
            for c in range(NUM_CLIENTS):
                ax.text(r, c, f"{mat[c, r]:.2f}", ha="center", va="center",
                        fontsize=6.5, color="black" if mat[c, r] < vmax * 0.65 else "white")
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.03, label="L2 drift")

    # ── Convergence-speed footnote inside panel 1 ────────────────────────
    experiments = [
        ("IID · FedAvg",              iid_pct,  purple),
        ("non-IID · FedAvg",          niid_pct, teal),
        (f"non-IID · FedProx µ={MU}", fp_pct,   orange),
    ]
    lines = []
    for name, pct, _ in experiments:
        r = _rounds_to(pct)
        lines.append(f"{name}: {'round ' + str(r) if r else 'not reached'}")
    ax1.text(0.03, 0.05, f"Rounds to {ACC_THRESHOLD:.0f}%:\n" + "\n".join(lines),
             transform=ax1.transAxes, fontsize=7.5, va="bottom",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

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

    iid_r,  iid_a,  iid_cd  = run_experiment(non_iid=False, label="Experiment 1 - IID  (uniform split)")
    niid_r, niid_a, niid_cd = run_experiment(non_iid=True,  label="Experiment 2 - non-IID (skewed split)")
    fp_r,   fp_a,   fp_cd   = run_experiment(non_iid=True,  label=f"Experiment 3 - non-IID + FedProx (µ={MU})", mu=MU)

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

    plot_results(iid_r, iid_a, iid_cd,
                 niid_r, niid_a, niid_cd,
                 fp_r,   fp_a,   fp_cd)
    print("\nDone! Check fl_results.png for the comparison chart.")
