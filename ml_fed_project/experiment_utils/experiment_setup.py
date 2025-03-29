import os
import yaml
import tensorflow as tf

def setup_experiment_directory(config):
    """Create experiment directory and return its path."""
    experiment_dir = os.path.join(os.getcwd(), 'experiments', config['experiment_name'])
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir

def save_config(config, experiment_dir):
    """Save configuration to the experiment directory."""
    config_path = os.path.join(experiment_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f) 

def create_directory_if_not_exists(directory_path):
    """
    Creates a directory if it doesn't already exist.
    
    Args:
        directory_path (str): Path to the directory to be created
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def setup_gpu():
    """Configure GPU memory growth"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)