import numpy as np
import tensorflow as tf
from typing import Any, Dict, List, Tuple
from pathlib import Path

from base.federated_client import FederatedClient
from config_utils.paths import ExperimentPaths
from model_utils import create_model,save_model,save_model_weights,load_model_from_disk,load_model_weights,save_training_history
from data_utils.data_loading import load_training_data
from model_utils import process_tf_history

class FedAvgClient(FederatedClient):
    def __init__(self, client_id: int, config: Dict[str, Any], experiment_paths: ExperimentPaths):
        """Initialize a FedAvg client.
        
        Args:
            client_id: Unique identifier for this client
            config: Configuration dictionary containing training parameters
            experiment_paths: ExperimentPaths object for managing file paths
        """
        self.client_id = client_id
        self.config = config
        self.experiment_paths = experiment_paths
        
        # Training parameters from config
        self.learning_rate = config.get('learning_rate', 0.01)
        self.local_epochs = config.get('local_epochs', 5)
        self.batch_size = config.get('batch_size', 32)
        
        # Initialize model and optimizer
        self.model = None
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        
        # Load or create local model
        self.load_or_create_local_model()

    def create_model(self) -> tf.keras.Model:
        """Create a fresh local model with initial architecture."""
        model = create_model(self.config['model_name'])
        return model

    def load_data(self, round_num: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load local data for a given round."""
        # Load data from the processed data directory
        data_path = self.experiment_paths.processed_data / f"client_{self.client_id}_round_{round_num}.npz"
        train_x, train_y = load_training_data(data_path)
        return train_x, train_y

    def train_on_local_data(self, train_x: np.ndarray, train_y: np.ndarray) -> Tuple[Dict[str, float], List[np.ndarray]]:
        """Perform local training on the provided data."""
        # Train the model
        history = self.model.fit(
            train_x, train_y,
            epochs=self.local_epochs,
            batch_size=self.batch_size,
            verbose=0
        )
        
        # Get final metrics
        metrics = process_tf_history(history)
        
        # Get updated weights
        updated_weights = self.model.get_weights()
        
        return metrics, updated_weights

    def load_or_create_local_model(self) -> None:
        """Load existing local model or create a new one if not found."""
        model_path = self.experiment_paths.models / f"client_{self.client_id}_model.keras"
        if model_path.exists():
            self.model = self._load_model_from_disk(str(model_path))
        else:
            self.model = self.create_model()

    def overwrite_weights_with_global(self, round_num: int) -> None:
        """Load and apply global weights from disk to local model."""
        global_weights_path = self.experiment_paths.global_model_path(round_num)
        if global_weights_path.exists():
            weights = self._load_weights_from_npz(str(global_weights_path))
            self.model.set_weights(weights)

    def train(self, round_num: int) -> Dict[str, float]:
        """High-level local training routine."""
        # Load global weights
        self.overwrite_weights_with_global(round_num)
        
        # Load local data
        train_x, train_y = self.load_data(round_num)
        
        # Train on local data
        metrics, updated_weights = self.train_on_local_data(train_x, train_y)
        
        # Save updated weights
        self.save_weights(round_num, updated_weights)
        
        # Save training history
        self._save_training_history(round_num, metrics)
        
        return metrics

    def save_weights(self, round_num: int, updated_weights: List[np.ndarray]) -> None:
        """Save updated weights to disk."""
        weights_path = self.experiment_paths.client_weights_path(self.client_id, round_num)
        save_model_weights(self.model,self.client_id,weights_path)

    def save_local_model(self) -> None:
        """Save the entire model to disk."""
        model_path = self.experiment_paths.models / f"client_{self.client_id}_model.keras"
        save_model(self.model,model_path)

    def _load_model_from_disk(self, model_path: str) -> tf.keras.Model:
        """Load a Keras model from disk."""
        return load_model_from_disk(model_path)

    def _load_weights_from_npz(self, npz_path: str) -> List[np.ndarray]:
        """Load weights from a .npz file."""
        return load_model_weights(npz_path)

    def _save_training_history(self, round_num: int, metrics: Dict[str, float]) -> None:
        """Save training history to disk."""
        history_path = self.experiment_paths.client_history_path(self.client_id, round_num)
        save_training_history(metrics,history_path)
    