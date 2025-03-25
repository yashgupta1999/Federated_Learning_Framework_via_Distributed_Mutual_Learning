import numpy as np
from typing import List, Dict

def average_weights(weights_list: List[Dict]) -> Dict:
    """
    Average a list of model weights (parameters).
    
    Args:
        weights_list (List[Dict]): List of model state dictionaries, where each dictionary
                                  contains parameter name as key and parameter tensor as value
    
    Returns:
        Dict: Averaged model weights
    """
    # Ensure we have weights to average
    if not weights_list:
        raise ValueError("Empty weights list provided")
    
    # Initialize the averaged weights with the first set of weights
    averaged_weights = {}
    for key in weights_list[0].keys():
        averaged_weights[key] = np.zeros_like(weights_list[0][key])
        
    # Sum up all weights
    for weights in weights_list:
        for key in weights.keys():
            averaged_weights[key] += weights[key]
    
    # Divide by number of models to get average
    n_models = len(weights_list)
    for key in averaged_weights.keys():
        averaged_weights[key] = averaged_weights[key] / n_models
        
    return averaged_weights
