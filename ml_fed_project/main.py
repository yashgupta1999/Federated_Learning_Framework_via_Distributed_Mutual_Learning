# from federated_frameworks.fed_avg import FederatedAveraging
from config_utils import load_config
from experiment_utils import setup_experiment_directory, save_config, create_directory_if_not_exists
from data_utils import split_data, process_and_save_folds, load_training_data
import os
from model_utils import create_model, train_model, save_model, load_model_from_disk,save_model_weights,load_all_weights,load_model_weights,clear_model_from_memory
from experiment_utils import setup_gpu
from federated_frameworks import run_federated_learning,save_history

def main():
    setup_gpu()
    # Load configuration settings from the YAML file
    config = load_config('config/fl_template_config.yaml')
    
    # Create and set up a directory for storing experiment results
    experiment_dir = setup_experiment_directory(config)
    print(experiment_dir)
    
    # Save the configuration settings to the experiment directory for reference
    save_config(config, experiment_dir)

    # Load & Distribute Data
    save_dir = os.path.join(os.getcwd(), 'experiments', config['experiment_name'], 'data')
    create_directory_if_not_exists(save_dir)

    data_path = os.path.join(os.getcwd(), 'data', config['data_path'])
    split_data(data_path, config['num_clients'], config['num_rounds'], save_dir, include_global=config['include_global'], stratified=config['stratified'])
    
    # #preprocess data
    data_dir = os.path.join(os.getcwd(), 'experiments', config['experiment_name'], 'data')
    process_and_save_folds(data_dir)


    fed_avg = run_federated_learning(config)

if __name__ == "__main__":
    main()