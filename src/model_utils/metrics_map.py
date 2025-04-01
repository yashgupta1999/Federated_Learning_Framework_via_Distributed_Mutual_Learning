import tensorflow as tf
from typing import List, Union

def is_tensorflow_model(model) -> bool:
    """
    Check if the given model is a TensorFlow model.
    
    Args:
        model: The model to check
        
    Returns:
        bool: True if the model is a TensorFlow model, False otherwise
    """
    return isinstance(model, (tf.keras.Model, tf.keras.Sequential))

def map_metrics_to_tf(metrics: Union[List[str], str]) -> List[Union[str, tf.keras.metrics.Metric]]:
    """
    Maps common metric names to TensorFlow metric objects.
    
    Args:
        metrics: A string or list of strings representing metric names
        
    Returns:
        List of TensorFlow metrics or metric names
    """
    # Convert single string to list for uniform processing
    if isinstance(metrics, str):
        metrics = [metrics]
    
    metrics_mapping = {
        'accuracy': tf.keras.metrics.Accuracy(),
        'binary_accuracy': tf.keras.metrics.BinaryAccuracy(),
        'categorical_accuracy': tf.keras.metrics.CategoricalAccuracy(),
        'precision': tf.keras.metrics.Precision(),
        'recall': tf.keras.metrics.Recall(),
        'auc': tf.keras.metrics.AUC(),
        'mae': tf.keras.metrics.MeanAbsoluteError(),
        'mse': tf.keras.metrics.MeanSquaredError(),
        'rmse': tf.keras.metrics.RootMeanSquaredError(),
        'sparse_categorical_accuracy': tf.keras.metrics.SparseCategoricalAccuracy(),
    }
    
    mapped_metrics = []
    for metric in metrics:
        metric = metric.lower()
        if metric in metrics_mapping:
            mapped_metrics.append(metrics_mapping[metric])
        else:
            # If no mapping exists, keep the original metric name
            mapped_metrics.append(metric)
            
    return mapped_metrics

def map_loss_to_tf(loss: Union[str, tf.keras.losses.Loss]) -> tf.keras.losses.Loss:
    """
    Maps common loss names to TensorFlow loss objects.
    
    Args:
        loss: A string or TensorFlow loss object representing the loss function
        
    Returns:
        TensorFlow loss object
    """ 
    loss_mapping = {
        'binary_crossentropy': tf.keras.losses.BinaryCrossentropy(),
        'categorical_crossentropy': tf.keras.losses.CategoricalCrossentropy(),
        'sparse_categorical_crossentropy': tf.keras.losses.SparseCategoricalCrossentropy(),
    }

    if isinstance(loss, str):
        loss = loss.lower()
        if loss in loss_mapping:
            return loss_mapping[loss]
        else:
            raise ValueError(f"Unsupported loss function: {loss}")
    return loss

def map_optimizer_to_tf(optimizer: Union[str, tf.keras.optimizers.Optimizer]) -> tf.keras.optimizers.Optimizer:
    """
    Maps common optimizer names to TensorFlow optimizer objects.
    
    Args:
        optimizer: A string or TensorFlow optimizer object representing the optimizer
        
    Returns:
        TensorFlow optimizer object
    """
    optimizer_mapping = {
        'adam': tf.keras.optimizers.Adam(),
        'sgd': tf.keras.optimizers.SGD(),
        'rmsprop': tf.keras.optimizers.RMSprop(),
    }   

    if isinstance(optimizer, str):
        optimizer = optimizer.lower()
        if optimizer in optimizer_mapping:
            return optimizer_mapping[optimizer]
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
    return optimizer



