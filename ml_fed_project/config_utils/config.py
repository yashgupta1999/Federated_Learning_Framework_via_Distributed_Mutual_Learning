from datetime import datetime
import os
import yaml
import string
import random
from .paths import ExperimentPaths

def load_config(config_path: str,) -> tuple[dict, ExperimentPaths]:
    """
    Load configuration and initialize paths.
    
    Args:
        config_path: Path to config YAML file
        experiment_name: Optional name for this experiment. If not provided, will use name from config file
                       or generate a new one if config doesn't exist.
    
    Returns:
        tuple of (config dict, ExperimentPaths object)
    """
    
    # if not os.path.exists(config_path):
    #     print(f"[INFO] Config file not found at {config_path}. Generating default config.")
        
    #     # Use default base config if none provided
    #     uid = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
    #     timestamp = datetime.now().strftime("%m%d%H%M")  # Optional short timestamp (MMDDHHMM)
    #     experiment_name = experiment_name or f"fl--{uid}-{timestamp}"
    #     base_config = {
    #         "experiment_name": experiment_name,
    #         "num_clients": 5,
    #         "num_rounds": 10,
    #         "local_epochs": 10,
    #         "image_size": (100, 100, 3),
    #         "random_seed": 333
    #     }
                        
    #     # Create directory if needed
    #     os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
    #     with open(config_path, 'w') as file:
    #         yaml.dump(base_config, file)
        
    #     # Initialize paths
    #     paths = ExperimentPaths(
    #         base_dir=base_config.get('base_dir'),
    #         experiment_name=experiment_name
    #     )
        
    #     # Save config to experiment directory
    #     paths.save_config(base_config)
        
    #     return base_config, paths

    # Load and return the config if it exists
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Use provided experiment_name if given, otherwise use from config
    final_experiment_name = config.get('experiment_name')
    if not final_experiment_name:
        raise ValueError("Experiment name must be provided either in config file")
    
    # Initialize paths
    paths = ExperimentPaths(
        base_dir=config.get('base_dir',),
        experiment_name=final_experiment_name
    )
    
    # Update config with final experiment name
    config['experiment_name'] = final_experiment_name
    
    # Save config to experiment directory
    paths.save_config(config)
    
    return config, paths