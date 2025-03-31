from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from numba import cuda
import gc
from tensorflow.keras.models import Model
import os
import numpy as np
from typing import Dict
from experiment_utils import create_directory_if_not_exists

from config_utils.config import load_config
from config_utils.paths import ExperimentPaths

def save_model(model, path):
    """Saves the given model to the specified path."""
    create_directory_if_not_exists(os.path.dirname(path))
    model.save(path)
    print(f"💾 Model saved to: {path}")

def clear_model_from_memory(model):
    """Deletes the model from memory and clears GPU session."""
    del model
    K.clear_session()
    gc.collect()
    # device = cuda.get_current_device()
    # device.reset()
    print("🧹 Model removed from memory and GPU session cleared.")

def load_model_from_disk(path):
    """Loads a model from the given path."""
    model = load_model(path)
    print(f"📦 Model loaded from: {path}")
    return model 

def save_model_weights(model: Model, client_id: str, folder_dir: str):
    """Save a model's weights to a .npy file in folder_dir."""
    create_directory_if_not_exists(folder_dir)
    os.makedirs(folder_dir, exist_ok=True)
    weights = model.get_weights()
    filepath = os.path.join(folder_dir, f"{client_id}_weights.npz")
    np.savez(filepath, *weights)
    # check if the file exists after saving
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Failed to save model weights for client {client_id} at: {filepath}")
    print(f"💾 Model weights saved for client {client_id} at: {filepath}")

def load_all_weights(folder_dir: str) -> Dict[str, list]:
    print(f"📂 Loading all model weights from: {folder_dir}")
    client_weights = {}
    for file in os.listdir(folder_dir):
        if file.endswith("_weights.npz"):
            client_id = file.replace("_weights.npz", "")
            data = np.load(os.path.join(folder_dir, file))
            weights = [data[f'arr_{i}'] for i in range(len(data.files))]
            client_weights[client_id] = weights
            print(f"📦 Loaded weights for client: {client_id}")
    print(f"✅ Successfully loaded weights for {len(client_weights)} clients")
    return client_weights

def load_model_weights(weight_path: str):
    print(f"📦 Loading model weights from: {weight_path}")
    data = np.load(weight_path)
    weights = [data[f'arr_{i}'] for i in range(len(data.files))]
    print("✅ Successfully loaded weights")
    return weights