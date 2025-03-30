import os
from pathlib import Path
from typing import Union, Optional
import yaml
from datetime import datetime

class ExperimentPaths:
    """Centralized path management for federated learning experiments."""
    
    def __init__(self, base_dir: Union[str, Path], experiment_name: str):
        """
        Initialize experiment paths structure.
        
        Args:
            base_dir: Root directory for all experiments
            experiment_name: Name of the specific experiment
        """
        # Add timestamp to experiment name for uniqueness
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = f"{experiment_name}"
        
        self.base = Path(base_dir) / 'experiments' / self.experiment_name
        
        # Core directories
        self.data_dir = self.base / 'data'
        self.models = self.base / 'models'
        self.weights = self.base / 'weights'
        self.history = self.base / 'history'
        self.processed_data = self.base / 'processed_data'
        self.logs = self.base / 'logs'
        self.metrics = self.base / 'metrics'
        self.config_file = self.base / 'config.yaml'
        
        # Create directories
        self._create_directories()
        
    def _create_directories(self) -> None:
        """Create all necessary directories if they don't exist."""
        dirs = [self.models, self.weights, self.history, 
                self.processed_data, self.logs, self.metrics, self.data_dir]
        for path in dirs:
            path.mkdir(parents=True, exist_ok=True)
    
    def client_weights_path(self, client_id: int, round_num: Optional[int] = None) -> Path:
        """Get path for client weights file."""
        client_dir = self.weights / f"client_{client_id}"
        client_dir.mkdir(exist_ok=True)
        
        if round_num is not None:
            return client_dir / f"round_{round_num}.npz"
        return client_dir / "latest.npz"
    
    def client_history_path(self, client_id: int, round_num: Optional[int] = None) -> Path:
        """Get path for client training history."""
        if round_num is not None:
            return self.history / f"client_{client_id}_round_{round_num}.pkl"
        return self.history / f"client_{client_id}_full.pkl"
    
    def metrics_path(self, round_num: Optional[int] = None) -> Path:
        """Get path for metrics CSV file."""
        if round_num is not None:
            return self.metrics / f"round_{round_num}_metrics.csv"
        return self.metrics / "all_metrics.csv"
    
    def global_model_path(self, round_num: Optional[int] = None) -> Path:
        """Get path for global model weights."""
        if round_num is not None:
            return self.models / f"global_model_round_{round_num}.h5"
        return self.models / "global_model_latest.h5"
    
    def log_path(self, name: str) -> Path:
        """Get path for log files."""
        return self.logs / f"{name}.log"
    
    def save_config(self, config: dict) -> None:
        """Save configuration to YAML file."""
        with open(self.config_file, 'w') as f:
            yaml.dump(config, f)
    
    def __str__(self) -> str:
        """String representation showing experiment name and base path."""
        return f"ExperimentPaths(experiment={self.experiment_name}, base={self.base})" 
    
    def get_data_dir(self) -> Path:
        """Get the data directory path."""
        return self.data_dir
    
    def get_base_dir(self) -> Path:
        """Get the base directory path."""
        return self.base