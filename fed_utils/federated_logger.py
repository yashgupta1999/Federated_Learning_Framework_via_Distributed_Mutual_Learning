import os
import json
import pandas as pd
from pathlib import Path
from model_utils import save_model_weights

class FederatedLogger:
    def __init__(self, config, experiment_path):
        self.config = config
        self.base_path = Path(experiment_path)
        self.logging_config = config['logging']

    def get_round_path(self, round_num):
        """Create and return path for specific round"""
        round_dir = self.base_path / f"{self.logging_config['directory_structure']['round_dir_prefix']}{round_num}"
        round_dir.mkdir(exist_ok=True)
        return round_dir

    def save_central_weights(self, model, round_num):
        """Save central model weights for the round"""
        round_path = self.get_round_path(round_num)
        weights_path = round_path / f"{self.logging_config['directory_structure']['central_weights_name']}.{self.logging_config['file_formats']['weights']}"
        save_model_weights(model, self.logging_config['directory_structure']['central_weights_name'], weights_path)

    def save_client_weights(self, client_id, model, round_num):
        """Save individual client weights"""
        round_path = self.get_round_path(round_num)
        weights_path = round_path / f"{self.logging_config['directory_structure']['client_weights_prefix']}{client_id}.{self.logging_config['file_formats']['weights']}"
        save_model_weights(model, self.logging_config['directory_structure']['client_weights_prefix'] + str(client_id), weights_path)

    def save_client_history(self, client_id, history, round_num):
        """Save training history for a client"""
        round_path = self.get_round_path(round_num)
        history_path = round_path / f"{self.logging_config['directory_structure']['history_prefix']}{client_id}.{self.logging_config['file_formats']['history']}"
        with open(history_path, 'w') as f:
            json.dump(history, f)

    def log_round_metrics(self, metrics_list, round_num):
        """Save metrics for the round in CSV format"""
        round_path = self.get_round_path(round_num)
        metrics_path = round_path / self.logging_config['directory_structure']['metrics_filename']
        df = pd.DataFrame(metrics_list)
        df.to_csv(metrics_path, index=False)

    def log_round(self, round_num, central_model, client_models, client_histories, metrics):
        """Log all information for a single round"""
        # Save central model
        self.save_central_weights(central_model, round_num)
        
        # Save client models and histories
        for client_id, (model, history) in enumerate(zip(client_models, client_histories), 1):
            self.save_client_weights(client_id, model, round_num)
            self.save_client_history(client_id, history, round_num)
        
        # Log metrics
        self.log_round_metrics(metrics, round_num)
