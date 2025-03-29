from datetime import datetime
import os
import yaml
import string
import random

def load_config(config_path, base_config=None):
    """
    Load a YAML config file. If it doesn't exist, create it with base_config.

    Parameters:
    - config_path (str): Path to the config YAML file
    - base_config (dict, optional): A dictionary representing the base config

    Returns:
    - config (dict): Loaded or newly created configuration
    """
    
    if not os.path.exists(config_path):
        print(f"[INFO] Config file not found at {config_path}. Generating default config.")
        
        # Use default base config if none provided
        if base_config is None:
            uid = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
            timestamp = datetime.now().strftime("%m%d%H%M")  # Optional short timestamp (MMDDHHMM)
            experiment_name = f"fl--{uid}-{timestamp}"
            base_config = {
                "experiment_name": experiment_name,
                "num_clients": 5,
                "num_rounds": 10,
                "local_epochs": 10,
                "image_size": (100, 100, 3),
                "random_seed": 333
            }
                        
        # Create directory if needed
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as file:
            yaml.dump(base_config, file)
        
        return base_config

    # Load and return the config if it exists
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config