import warnings
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import gc
import random
from tensorflow.keras import backend as K
from keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from config import CONFIG
import yaml
import ast
from tensorflow.keras.models import load_model
from numba import cuda 

def create_model():
    """
    Creates and compiles a CNN model based on the provided configuration.

    Parameters:
    - config (dict): Configuration dictionary with keys:
        - 'random_seed': int
        - 'image_size': str (e.g., '(100, 100, 3)')
    
    Returns:
    - model (tf.keras.Model): Compiled Keras model
    """
    # Set random seed for reproducibility
    seed = CONFIG.get('random_seed', 333)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
   
    # Parse image input shape from string
    input_shape = ast.literal_eval(CONFIG.get('image_size', '(100, 100, 3)'))
    # Build model
    model = Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])

    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.Recall(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.AUC(
                num_thresholds=200,
                curve='ROC',
                summation_method='interpolation'
            ),
            'accuracy'
        ]
    )

    return model

def train_model(model, trainx, trainy, epochs=CONFIG.get("local_epochs", 10), batch_size=16, verbose=1, validation_data=None):
    """
    Trains a compiled Keras model.

    Args:
        model (tf.keras.Model): A compiled Keras model.
        trainx (np.ndarray): Training input data.
        trainy (np.ndarray): Training labels.
        epochs (int): Number of training epochs.
        batch_size (int): Size of training batches.
        verbose (int): Verbosity mode (0=silent, 1=progress bar, 2=one line per epoch).
        validation_data (tuple): Optional (val_x, val_y) for validation.

    Returns:
        tf.keras.callbacks.History: Training history object.
    """
    history = model.fit(
        trainx,
        trainy,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        validation_data=validation_data
    )
    return history
    
def save_model(model, path):
    """
    Saves the given model to the specified path.

    Args:
        model (tf.keras.Model): The model to save.
        path (str): Path where the model will be saved.
    """
    model.save(path)
    print(f"💾 Model saved to: {path}")

def clear_model_from_memory(model):
    """
    Deletes the model from memory and clears GPU session.

    Args:
        model (tf.keras.Model): The model to delete.
    """
    del model
    K.clear_session()
    gc.collect()
    device = cuda.get_current_device()
    device.reset()
    print("🧹 Model removed from memory and GPU session cleared.")

def load_model_from_disk(path):
    """
    Loads a model from the given path.

    Args:
        path (str): Path to the saved model.

    Returns:
        tf.keras.Model: Loaded model.
    """
    model = load_model(path)
    print(f"📦 Model loaded from: {path}")
    return model
