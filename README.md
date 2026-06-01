# Federated Learning Demo
**Flower + PyTorch · FedAvg · IID vs non-IID comparison**

A full federated learning simulation showing 5 clients training a CNN on private
local data shards — raw data never leaves each client, only model weights are shared.

## What it does
- Simulates 5 clients each holding a private, non-overlapping data shard
- Implements FedAvg (McMahan et al. 2017) for server-side aggregation
- Compares IID (uniform) vs non-IID (skewed) data distributions
- Produces a comparison chart showing convergence gap

## Quickstart
```bash
pip install -r requirements.txt
python simulate.py
```

## Project structure
```
├── model.py        # CNN architecture + weight helpers
├── data.py         # Data loading + IID/non-IID partitioning
├── client.py       # Flower NumPyClient
├── server.py       # FedAvg strategy + aggregation helpers
├── simulate.py     # Orchestrator — runs all experiments
├── run_client.py   # Subprocess entry point per client
└── Dockerfile
```

## Key results
| Mode    | Final accuracy | Convergence speed |
|---------|---------------|-------------------|
| IID     | ~98%          | Fast (round 1)    |
| non-IID | ~97%          | Slow (3+ rounds)  |

The 1-2% gap and slower early convergence under non-IID is the core FL research
finding this demo reproduces — it motivates techniques like FedProx and SCAFFOLD.

## Resume bullet points
```
• Simulated 5-client federated training with non-IID data partitioning (Flower + PyTorch)
• Implemented FedAvg aggregation; global model reached 97%+ accuracy without raw data sharing
• Benchmarked IID vs non-IID convergence — reproduced accuracy gap matching FL literature
• Containerized with Docker; reproducible in one command
```
