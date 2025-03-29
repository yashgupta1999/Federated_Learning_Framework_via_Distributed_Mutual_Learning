import numpy as np
from typing import List, Dict
from model_utils import load_all_weights

def fed_avg_from_disk(folder_dir: str, client_scores: Dict[str, float]):
    """
    Load all weights from folder_dir, perform FedAvg, and update the model.
    - folder_dir: Directory containing client weights
    - client_scores: Dict mapping client_id to weight (e.g., sample count)
    """
    client_weights = load_all_weights(folder_dir)
    total_score = sum(client_scores.values())
    weighted_avg = None

    for client_id, weights in client_weights.items():
        score = client_scores.get(client_id, 0) / total_score
        if weighted_avg is None:
            weighted_avg = [score * layer for layer in weights]
        else:
            for i in range(len(weights)):
                weighted_avg[i] += score * weights[i]

    return weighted_avg
