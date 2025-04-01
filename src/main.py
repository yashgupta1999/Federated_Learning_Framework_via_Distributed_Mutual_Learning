# from federated_frameworks.fed_avg import FederatedAveraging
from config_utils import load_config, ExperimentPaths
from data_utils import split_data, process_and_save_folds
from experiment_utils import setup_gpu
from federated_frameworks import FedAvgClient

def main():

    setup_gpu()
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

if __name__ == "__main__":
    main()