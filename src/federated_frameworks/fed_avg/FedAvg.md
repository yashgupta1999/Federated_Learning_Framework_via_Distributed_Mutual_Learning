# FedAvg Algorithm

## Overview

FedAvg (Federated Averaging) is a widely used algorithm in federated learning that enables training a global model on decentralized data residing on multiple client devices. In FedAvg, each client performs local training on its own data and sends its model updates to a central server, which then aggregates these updates by averaging to obtain a new global model.

## Motivation

Federated learning is designed to overcome privacy and communication challenges by keeping raw data on local devices. FedAvg facilitates this by:

- **Privacy Preservation:** Clients share model updates rather than raw data.
- **Communication Efficiency:** Local updates are aggregated periodically rather than transmitting data continuously.
- **Scalability:** The approach scales to many clients with heterogeneous data distributions, making it suitable for real-world applications like mobile devices.

## How FedAvg Works

### Global Training Loop (Server-Side)

1. **Broadcast:** The server initializes a global model \( w^0 \) and sends it to a selected subset of clients.
2. **Local Update:** Each client updates the received model by training on its local data for several epochs using an optimizer like SGD.
3. **Aggregation:** Clients send their updated models (or the differences) back to the server, which aggregates them (typically via weighted averaging based on local dataset sizes) to update the global model.
4. **Iteration:** The process is repeated over several communication rounds until the global model converges.

### Local Training on Clients

Each client solves a standard optimization problem on its local data:

\[
\min_{w} \; f_i(w)
\]

- \( f_i(w) \) represents the local loss on client \( i \).
- Clients perform multiple epochs of training using their local dataset and update the model accordingly.

## Pseudocode

Below is the pseudocode outlining the FedAvg algorithm:

```plaintext
Algorithm FedAvg
Input: Global model w₀, number of rounds T, client learning rate η, number of local epochs E, set of clients C

for t = 0 to T-1 do
    Server broadcasts the current global model wᵗ to a subset of clients S ⊆ C
    for each client i ∈ S do in parallel
        wᵢ = LocalUpdate(wᵗ, fᵢ, η, E)
    end for
    // Aggregate updates from clients, typically weighted by local dataset sizes
    wᵗ⁺¹ = Aggregate({wᵢ | i ∈ S})
end for

Function LocalUpdate(w, f, η, E)
    Input: Global model w, local loss function f, learning rate η, number of local epochs E
    Initialize: w_local = w
    for epoch = 1 to E do
        for each minibatch B from the client’s local data do
            // Compute the gradient of the local loss
            g = ∇f_B(w_local)
            // Update local model
            w_local = w_local - η * g
        end for
    end for
    return w_local
