from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
import numpy as np
import tensorflow as tf
import os

class FederatedClient(ABC):
    @abstractmethod
    def __init__(self, client_id: int, config: Dict[str, Any], experiment_paths: Dict[str, str]):
        """Initialize a federated learning client.
        
        Args:
            client_id: Unique identifier for this client
            config: Configuration dictionary containing global and client-specific settings
            experiment_paths: Dictionary of directory paths for models, weights, data, etc.
        """
        pass

    @abstractmethod
    def create_model(self) -> Any:
        """Create a fresh local model with initial architecture.
        
        Returns:
            A model instance (e.g., tf.keras.Model)
        """
        pass

    @abstractmethod
    def load_data(self, round_num: int) -> Tuple[Any, Any]:
        """Load local data for a given round from disk.
        
        Args:
            round_num: The round number for which data should be loaded
            
        Returns:
            Tuple of (train_x, train_y) or a data loader object
        """
        pass

    @abstractmethod
    def train_on_local_data(self, train_x: Any, train_y: Any) -> Tuple[Dict[str, float], List[np.ndarray]]:
        """Perform local training on the provided data.
        
        Args:
            train_x: Training features
            train_y: Training labels
            
        Returns:
            Tuple containing:
            1. metrics: Dictionary of local metrics (loss, accuracy, num_samples, etc.)
            2. updated_weights: List of numpy arrays representing the trained model's weights
        """
        pass

    @abstractmethod
    def load_or_create_local_model(self) -> None:
        """Load existing local model or create a new one if not found."""
        pass

    @abstractmethod
    def overwrite_weights_with_global(self, round_num: int) -> None:
        """Load and apply global weights from disk to local model."""
        pass

    @abstractmethod
    def train(self, round_num: int) -> Dict[str, float]:
        """High-level local training routine.
        
        Args:
            round_num: Current training round number
            
        Returns:
            Dictionary of training metrics
        """
        pass

    @abstractmethod
    def save_weights(self, round_num: int, updated_weights: List[np.ndarray]) -> None:
        """Save updated weights to disk.
        
        Args:
            round_num: Current round number
            updated_weights: List of numpy arrays to save
        """
        pass

    @abstractmethod
    def save_local_model(self) -> None:
        """Save the entire model to disk."""
        pass

    @abstractmethod
    def _load_model_from_disk(self, model_path: str) -> Any:
        """Load a Keras model from disk.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Loaded Keras model
        """
        pass

    @abstractmethod
    def _load_weights_from_npz(self, npz_path: str) -> List[np.ndarray]:
        """Load weights from a .npz file.
        
        Args:
            npz_path: Path to the weights file
            
        Returns:
            List of numpy arrays representing the weights
        """
        pass 