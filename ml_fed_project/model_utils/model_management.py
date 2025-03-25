from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from numba import cuda
import gc

def save_model(model, path):
    """Saves the given model to the specified path."""
    model.save(path)
    print(f"💾 Model saved to: {path}")

def clear_model_from_memory(model):
    """Deletes the model from memory and clears GPU session."""
    del model
    K.clear_session()
    gc.collect()
    device = cuda.get_current_device()
    device.reset()
    print("🧹 Model removed from memory and GPU session cleared.")

def load_model_from_disk(path):
    """Loads a model from the given path."""
    model = load_model(path)
    print(f"📦 Model loaded from: {path}")
    return model 