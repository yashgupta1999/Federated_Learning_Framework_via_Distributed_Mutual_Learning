# process_tf_history.py

import tensorflow as tf
from typing import Dict, Any, List, Union
from model_utils.training_utils import map_metrics_to_tf

def process_tf_history(
    config: Dict[str, Any], 
    tf_history: tf.keras.callbacks.History
) -> Dict[str, List[float]]:
    """
    Process TensorFlow history object into a dictionary of metrics.
    
    Args:
        config: Configuration dictionary containing model metrics under 'model_metrics' key
        tf_history: TensorFlow history object containing training metrics
        
    Returns:
        Dictionary mapping metric names to their history values as lists of floats
        
    Raises:
        KeyError: If 'model_metrics' key is not found in config
        ValueError: If tf_history is empty or metrics are not found
    """
    if not isinstance(tf_history.history, dict) or not tf_history.history:
        raise ValueError("TensorFlow history object is empty or invalid")
        
    if 'model_metrics' not in config:
        raise KeyError("'model_metrics' not found in config dictionary")
    
    metrics = config['model_metrics']
    if not metrics:
        raise ValueError("No metrics specified in config['model_metrics']")
    
    history: Dict[str, List[float]] = {}
    
    for metric in metrics:
        if metric not in tf_history.history:
            raise ValueError(f"Metric '{metric}' not found in training history")
        history[metric] = tf_history.history[metric]
    
    return history

