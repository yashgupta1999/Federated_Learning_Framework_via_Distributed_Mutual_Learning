from abc import ABC, abstractmethod
from typing import Any, Dict, List
import numpy as np
import os
from .federated_client import FederatedClient
from config_utils.config import ExperimentPaths

class FederatedServer(ABC):
    def __init__(self, config: Dict[str, Any], experiment_paths: ExperimentPaths, clients: List[FederatedClient]):
        """Initialize a federated learning server.
        
        Args:
            config: Global configuration dictionary
            experiment_paths: ExperimentPaths object for managing experiment directories
            clients: List of all possible client objects
        """
        self.config = config
        self.paths = experiment_paths
        self.clients = clients
        self.history = {}

    @abstractmethod
    def initialize_global_model(self) -> None:
        """Create or load the global model and save initial weights to disk."""
        pass

    @abstractmethod
    def select_clients(self, round_num: int) -> List[FederatedClient]:
        """Select which clients will participate in this round.
        
        Args:
            round_num: Current training round number
            
        Returns:
            List of selected FederatedClient instances
        """
        pass

    @abstractmethod
    def aggregate(self, client_weight_paths: List[str], 
                 client_metrics: Dict[int, Dict[str, float]], 
                 round_num: int) -> List[np.ndarray]:
        """Aggregate client weights to create new global weights.
        
        Args:
            client_weight_paths: List of paths to client weight files
            client_metrics: Dictionary mapping client IDs to their training metrics
            round_num: Current round number
            
        Returns:
            List of numpy arrays representing the new global weights
        """
        pass

    @abstractmethod
    def run(self) -> None:
        """Main federated training loop."""
        pass

    @abstractmethod
    def finalize(self) -> None:
        """Perform any necessary cleanup or final logging."""
        pass

    @abstractmethod
    def log_round_metrics(self, round_num: int, metrics: Dict[str, Any]) -> None:
        """Log metrics for the current round.
        
        Args:
            round_num: Current round number
            metrics: Dictionary of metrics to log
        """
        pass

    @abstractmethod
    def _parse_client_id_from_path(self, path: str) -> int:
        """Extract client ID from a file path.
        
        Args:
            path: File path containing client ID (e.g., 'client_2_round5.npz')
            
        Returns:
            Extracted client ID
        """
        pass 