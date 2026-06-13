# Federated Learning — FedAvg & FedProx from scratch

A self-contained federated learning simulation built directly on PyTorch (no FL
framework). It implements **FedAvg** and **FedProx** by hand and instruments
**per-client weight drift** to show *why* non-IID client data slows convergence —
and how FedProx's proximal term mitigates it.

> The point of this repo is the **mechanics and the diagnostics**, not a headline
> accuracy number. The dataset is synthetic and deliberately separable (see below),
> so accuracy is not a meaningful benchmark — the weight-drift behaviour is.

## What it implements

- **FedAvg** (McMahan et al., 2017) — sample-count-weighted aggregation of client weights.
- **FedProx** (Li et al., 2020) — adds a proximal term `µ/2 · ‖w − w_global‖²` to each
  client's local loss to keep local updates close to the global model.
- **IID vs non-IID partitioning** — 5 clients, non-overlapping shards; non-IID clients
  are each dominated by 2 of 10 classes.
- **Per-client L2 weight drift tracking** — measures how far each client pulls away from
  the global model per round, visualised as per-round curves and per-client heatmaps.

Three experiments are run and compared: IID·FedAvg, non-IID·FedAvg, non-IID·FedProx.

## The dataset (read this before trusting any number)

`data.py` generates a synthetic 10-class set: each class is a Gaussian cluster with a
distinct, near-orthogonal mean (`SIGNAL=2.5`, `NOISE=1.5`). This is **near
linearly separable by construction** — high accuracy is expected and is *not* evidence
of model quality. The signal/noise ratio was chosen specifically so the IID vs non-IID
convergence gap is visible within a few rounds. It is a controlled illustration, not a
real-data result. Use a real federated dataset (e.g. partitioned MNIST/CIFAR or LEAF)
if you want a benchmark.

## Quickstart

```bash
pip install -r requirements.txt
python simulate.py          # runs all 3 experiments, writes fl_results.png
```

`requirements.txt`:

```
torch
numpy
matplotlib
```

Or with Docker:

```bash
docker build -t fl-demo .
docker run --rm -v "$PWD:/out" fl-demo   # chart written to fl_results.png in /app
```

## Project structure

```
├── simulate.py   # entry point — from-scratch FedAvg/FedProx, runs experiments, plots
├── model.py      # SimpleNet (MLP) + state_dict <-> ndarray helpers
├── data.py       # synthetic dataset + IID / non-IID partitioning
├── Dockerfile
└── fl_results.png
```

## What the chart shows (`fl_results.png`)

1. **Accuracy over rounds** — IID·FedAvg vs non-IID·FedAvg vs non-IID·FedProx, with the
   convergence-threshold annotation (rounds to reach the threshold).
2. **Mean L2 weight drift per round** — how heterogeneity manifests as larger client
   weight deltas, and how FedProx suppresses them.
3–4. **Per-client drift heatmaps** — FedAvg vs FedProx, showing which clients drift most
   and how the proximal term flattens the distribution.

## What this demonstrates

- A correct, framework-free implementation of FedAvg and FedProx.
- A working measurement of client heterogeneity (weight drift), not just accuracy.
- The expected qualitative effect: non-IID data increases per-client drift and slows
  early convergence; FedProx's proximal term reduces drift toward the global model.

## Notes / honest limitations

- Synthetic, separable data — accuracy is illustrative only (see *The dataset*).
- The IID/non-IID gap magnitude depends on the tuned signal/noise ratio.
- Single-process simulation (clients run sequentially), not real distributed training.
- No secure aggregation, differential privacy, or communication-cost modelling.

## References

- McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data*, 2017.
- Li et al., *Federated Optimization in Heterogeneous Networks* (FedProx), 2020.
