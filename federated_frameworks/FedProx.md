# FedProx Algorithm

## Overview

FedProx is an extension of the popular federated averaging algorithm (FedAvg). It was developed to address challenges in federated learning, particularly when client data is non-IID (not identically distributed) or when clients have varying computational capabilities. By incorporating a proximal term in the local objective function, FedProx ensures that local updates remain close to the current global model, resulting in more stable and robust convergence.

## Motivation

Federated learning often involves training on decentralized and heterogeneous data. In traditional FedAvg, each client performs local updates based solely on its own data. However, when the data distribution varies significantly between clients, local models may diverge from one another, potentially slowing down convergence or reducing model performance.

FedProx addresses these issues by adding a proximal term to the local loss function. This term penalizes deviations from the global model, which:

- **Mitigates Data Heterogeneity:** It reduces the impact of non-IID data by keeping local models closer to the global model.
- **Enhances Stability:** It prevents any single client’s update from straying too far, thereby improving the consistency of aggregated updates.
- **Accommodates System Heterogeneity:** It allows clients with different computational resources (and possibly varying numbers of local training epochs) to contribute effectively to the overall training process.

## How FedProx Works

### Global Training Loop (Server-Side)

1. **Broadcast:** The server starts with an initial global model \( w^0 \) and broadcasts it to a selected subset of clients.
2. **Local Update:** Each client performs local training on its dataset, minimizing a modified objective function that includes a proximal term.
3. **Aggregation:** The server collects the updated models from the clients and aggregates them (typically via a weighted average) to form a new global model.
4. **Iteration:** The process repeats over several communication rounds until the global model converges.

### Local Objective with Proximal Term

For a given client \( i \), the local objective becomes:

\[
\min_{w} \; f_i(w) + \frac{\mu}{2}\|w - w^t\|^2
\]

- \( f_i(w) \) is the local loss function.
- \( w^t \) is the current global model at round \( t \).
- \( \mu \) is a hyperparameter controlling the strength of the proximal term.

The proximal term \( \frac{\mu}{2}\|w - w^t\|^2 \) serves as a regularizer that penalizes large deviations from the global model, thereby anchoring the local updates.

## Pseudocode

Below is the pseudocode outlining the FedProx algorithm:

```plaintext
Algorithm FedProx
Input: Global model w₀, number of rounds T, proximal parameter μ, client learning rate η, number of local epochs E, set of clients C

for t = 0 to T-1 do
    Server broadcasts the current global model wᵗ to a subset of clients S ⊆ C
    for each client i ∈ S do in parallel
        wᵢ = LocalUpdate(wᵗ, fᵢ, μ, η, E)
    end for
    // Aggregate updates from clients, typically weighted by local dataset sizes
    wᵗ⁺¹ = Aggregate({wᵢ | i ∈ S})
end for

Function LocalUpdate(w, f, μ, η, E)
    Input: Global model w, local loss function f, proximal parameter μ, learning rate η, number of local epochs E
    Initialize: w_local = w
    for epoch = 1 to E do
        for each minibatch B from the client’s local data do
            // Compute the gradient of the local loss plus the proximal term
            g = ∇f_B(w_local) + μ * (w_local - w)
            // Update local model
            w_local = w_local - η * g
        end for
    end for
    return w_local
