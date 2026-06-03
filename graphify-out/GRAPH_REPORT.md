# Graph Report - .  (2026-06-03)

## Corpus Check
- Corpus is ~2,120 words - fits in a single context window. You may not need a graph.

## Summary
- 58 nodes · 113 edges · 13 communities (10 shown, 3 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 8 edges (avg confidence: 0.88)
- Token cost: 4,800 input · 2,100 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Non-IID Data Partitioning|Non-IID Data Partitioning]]
- [[_COMMUNITY_FedAvg Server Strategy|FedAvg Server Strategy]]
- [[_COMMUNITY_Client Model Parameters|Client Model Parameters]]
- [[_COMMUNITY_Flower Client Interface|Flower Client Interface]]
- [[_COMMUNITY_Neural Network and Evaluation|Neural Network and Evaluation]]
- [[_COMMUNITY_Simulation and Plotting|Simulation and Plotting]]
- [[_COMMUNITY_Experiment Orchestration|Experiment Orchestration]]
- [[_COMMUNITY_Flower Fit Contract|Flower Fit Contract]]
- [[_COMMUNITY_Federated Learning Concepts|Federated Learning Concepts]]
- [[_COMMUNITY_Client Drift Analysis|Client Drift Analysis]]
- [[_COMMUNITY_FedProx Algorithm|FedProx Algorithm]]
- [[_COMMUNITY_FedAvg Simulation|FedAvg Simulation]]
- [[_COMMUNITY_Local Training Function|Local Training Function]]

## God Nodes (most connected - your core abstractions)
1. `run_experiment()` - 17 edges
2. `load_partition()` - 10 edges
3. `get_parameters()` - 10 edges
4. `FlowerClient` - 7 edges
5. `build_strategy()` - 7 edges
6. `describe_partition()` - 6 edges
7. `SimpleNet` - 6 edges
8. `set_parameters()` - 6 edges
9. `train_client()` - 6 edges
10. `train_client_fedprox()` - 6 edges

## Surprising Connections (you probably didn't know these)
- `build_strategy()` --semantically_similar_to--> `fedavg()`  [INFERRED] [semantically similar]
  server.py → simulate.py
- `Data Privacy via Weight Sharing` --rationale_for--> `FedAvg Algorithm (McMahan et al. 2017)`  [INFERRED]
  README.md → simulate.py
- `Federated Learning Demo (README)` --references--> `FedAvg Algorithm (McMahan et al. 2017)`  [EXTRACTED]
  README.md → simulate.py
- `Federated Learning Demo (README)` --references--> `Non-IID Data Partitioning`  [EXTRACTED]
  README.md → data.py
- `Non-IID Data Partitioning` --conceptually_related_to--> `Client Weight Drift (L2)`  [INFERRED]
  data.py → simulate.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Federated Training Round: Local Training → Aggregation → Global Update** — simulate_train_client, simulate_train_client_fedprox, simulate_fedavg, model_set_parameters, simulate_evaluate [EXTRACTED 0.95]
- **Flower Client–Server FL Contract (NumPyClient interface)** — client_flowerclient, client_flowerclient_fit, client_flowerclient_evaluate, server_build_strategy, server_weighted_average [EXTRACTED 0.95]
- **Non-IID Challenge and Mitigation Stack** — concept_non_iid_data, concept_client_drift, concept_fedprox, simulate_train_client_fedprox, data_load_partition [INFERRED 0.85]

## Communities (13 total, 3 thin omitted)

### Community 0 - "Non-IID Data Partitioning"
Cohesion: 0.43
Nodes (7): _build(), describe_partition(), load_partition(), _make_means(), bool, int, Synthetic 10-class dataset for the federated learning demo.  Design:   - Each cl

### Community 1 - "FedAvg Server Strategy"
Cohesion: 0.38
Nodes (6): FedAvg, Metrics, build_strategy(), int, Weighted average of accuracy across all clients., weighted_average()

### Community 2 - "Client Model Parameters"
Cohesion: 0.60
Nodes (3): get_parameters(), Compact MLP for the 64-dim synthetic feature dataset., set_parameters()

### Community 3 - "Flower Client Interface"
Cohesion: 0.40
Nodes (3): bool, float, int

### Community 4 - "Neural Network and Evaluation"
Cohesion: 0.40
Nodes (3): _TESTLOADER (global test DataLoader), SimpleNet, evaluate()

### Community 5 - "Simulation and Plotting"
Cohesion: 0.50
Nodes (4): plot_results(), Federated Learning Demo — PyTorch + FedAvg / FedProx ===========================, First round index (1-based) where accuracy crosses threshold, else None., _rounds_to()

### Community 6 - "Experiment Orchestration"
Cohesion: 0.40
Nodes (5): bool, float, Returns (rounds, accs, client_drifts) where client_drifts[round][client] = L2 dr, run_experiment(), str

### Community 8 - "Federated Learning Concepts"
Cohesion: 0.67
Nodes (4): Data Privacy via Weight Sharing, FedAvg Algorithm (McMahan et al. 2017), Non-IID Data Partitioning, Federated Learning Demo (README)

### Community 9 - "Client Drift Analysis"
Cohesion: 0.67
Nodes (3): Client Weight Drift (L2), _l2_drift(), L2 norm of weight delta between two parameter lists.

### Community 10 - "FedProx Algorithm"
Cohesion: 0.67
Nodes (3): FedProx Algorithm (Li et al. 2020), FedProx local step: adds µ/2 * ||w - w_global||² to the loss., train_client_fedprox()

## Knowledge Gaps
- **8 isolated node(s):** `bool`, `float`, `Metrics`, `FedAvg`, `bool` (+3 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `run_experiment()` connect `Experiment Orchestration` to `Non-IID Data Partitioning`, `Client Model Parameters`, `Neural Network and Evaluation`, `Simulation and Plotting`, `Federated Learning Concepts`, `Client Drift Analysis`, `FedProx Algorithm`, `FedAvg Simulation`, `Local Training Function`?**
  _High betweenness centrality (0.292) - this node is a cross-community bridge._
- **Why does `get_parameters()` connect `Client Model Parameters` to `FedAvg Server Strategy`, `Simulation and Plotting`, `Experiment Orchestration`, `Flower Fit Contract`, `FedProx Algorithm`, `Local Training Function`?**
  _High betweenness centrality (0.149) - this node is a cross-community bridge._
- **Why does `build_strategy()` connect `FedAvg Server Strategy` to `Client Model Parameters`, `FedAvg Simulation`, `Neural Network and Evaluation`?**
  _High betweenness centrality (0.137) - this node is a cross-community bridge._
- **What connects `bool`, `float`, `Synthetic 10-class dataset for the federated learning demo.  Design:   - Each cl` to the rest of the system?**
  _18 weakly-connected nodes found - possible documentation gaps or missing edges._