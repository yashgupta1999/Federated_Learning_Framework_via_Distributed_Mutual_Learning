# from federated_frameworks.fed_avg import FederatedAveraging
from config_utils import load_config, ExperimentPaths
from experiment_utils import setup_experiment_directory, save_config, create_directory_if_not_exists
from data_utils import split_data, process_and_save_folds, load_training_data
import os
from model_utils import create_model, train_model, save_model, load_model_from_disk,save_model_weights,load_all_weights,load_model_weights,clear_model_from_memory
from experiment_utils import setup_gpu
from federated_frameworks import run_federated_learning,save_history

def main():
    # setup_gpu()
    # Load configuration settings from the YAML file
    config, experiment_paths = load_config('config/fl_template_config.yaml')

    split_data(
        config['data_path'],
        config['num_clients'], 
        config['num_rounds'], 
        experiment_paths.data_dir, 
        include_global=config['include_global'], 
        stratified=config['stratified']
    )
    
    # # Preprocess data
    process_and_save_folds(experiment_paths.data_dir,experiment_paths.processed_data)

    fed_avg = run_federated_learning(config,experiment_paths)

if __name__ == "__main__":
    main()