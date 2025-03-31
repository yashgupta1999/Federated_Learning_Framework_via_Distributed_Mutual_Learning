import sys
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from data_utils import load_training_data
from config_utils.paths import ExperimentPaths
from model_utils import (
    create_model, 
    train_model, 
    save_model, 
    save_model_weights, 
    load_model_from_disk,
    load_model_weights,
    clear_model_from_memory
)

from fed_utils import fed_avg_from_disk, FederatedLogger
import pickle
from datetime import datetime
from colorama import Fore, Style, init
from experiment_utils import create_directory_if_not_exists
init(autoreset=True)  # Initialize colorama

def setup_gpu():
    """Configure GPU memory growth"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

def log_section(title):
    """Print a formatted section header"""
    print(f"\n{Fore.CYAN}{'='*80}")
    print(f"{Fore.CYAN}== {title}")
    print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}\n")

def log_subsection(title):
    """Print a formatted subsection header"""
    print(f"\n{Fore.YELLOW}{'-'*60}")
    print(f"{Fore.YELLOW}-- {title}")
    print(f"{Fore.YELLOW}{'-'*60}{Style.RESET_ALL}\n")

def initialize_central_model(config, experiment_paths: ExperimentPaths, history_dict=None):
    """Initialize and train the central model"""
    log_section("Initializing Central Model")
    print(f"{Fore.GREEN}• Creating central model architecture...")
    
    if history_dict is None:
        history_dict = {}
    
    model = create_model(config['seed'])
    data_path = os.path.join(experiment_paths.processed_data, 'init.npy')
    print(f"{Fore.GREEN}• Loading initial training data from: {data_path}")
    trainx, trainy = load_training_data(data_path)
    print(f"{Fore.GREEN}• Training central model with {len(trainx)} samples...")
    
    history = train_model(model, trainx, trainy, epochs=config['local_epochs'])
    history_dict['init'] = history
    
    # Save central model
    model_path = os.path.join(experiment_paths.models, 'central_model.keras')
    weights_path = os.path.join(experiment_paths.weights)
    
    save_model(model, model_path)
    save_model_weights(model, 'central_model', weights_path)
    clear_model_from_memory(model)
    
    print(f"{Fore.GREEN}• Saving central model and weights...")
    
    print(f"{Fore.GREEN}✓ Central model initialization complete!")
    return history_dict

def initialize_client_models(config, history_dict=None, experiment_paths: ExperimentPaths = None):
    """Initialize models for all clients"""
    log_section("Initializing Client Models")
    
    if history_dict is None:
        history_dict = {}
        
    print(f"{Fore.GREEN}• Creating {config['num_clients']} client models...")
    for i in range(config['num_clients']):
        client_name = f'client_{i+1}'
        print(f"{Fore.WHITE}  - Initializing {client_name}...")
        model = create_model(seed=config['seed']+i+1)
        model._name = client_name
        
        model_path = os.path.join(experiment_paths.models, f'{client_name}.keras')
        save_model(model, model_path)
        history_dict[client_name] = {}
        clear_model_from_memory(model)
    
    print(f"{Fore.GREEN}✓ All client models initialized successfully!")
    return history_dict

def train_client(client_num, round_num, central_weights, config, history_dict, experiment_paths: ExperimentPaths):
    """Train an individual client"""
    client_name = f'client_{client_num}'
    log_subsection(f"Training {client_name} - Round {round_num}")
    
    # Initialize client's dictionary if it doesn't exist
    if client_name not in history_dict:
        history_dict[client_name] = {}
    
    print(f"{Fore.WHITE}• Loading model and setting weights...")
    model_path = os.path.join(experiment_paths.models, f'client_{client_num}.keras')
    model = load_model_from_disk(model_path)
    model.set_weights(central_weights)
    
    print(f"{Fore.WHITE}• Loading training data...")
    data_path = os.path.join(experiment_paths.processed_data, f'client_{client_num}', f'round{round_num}.npy')
    trainx, trainy = load_training_data(data_path)
    print(f"{Fore.WHITE}• Training with {len(trainx)} samples for {config['local_epochs']} epochs...")
    
    history = train_model(model, trainx, trainy, epochs=config['local_epochs'])
    
    print(f"{Fore.WHITE}• Saving model and weights...")
    history_dict[client_name][f'round_{round_num}'] = history
    save_model(model, model_path)
    
    weights_path = os.path.join(experiment_paths.weights, 'client_weights')
    save_model_weights(model, client_name, weights_path)
    
    clear_model_from_memory(model)
    
    print(f"{Fore.GREEN}✓ Client training complete!")
    return {client_name: len(trainx)}, history_dict

def train_round(round_num, config, history_dict):
    """Execute one round of federated training"""
    log_section(f"Training Round {round_num}/{config['num_rounds']}")
    
    print(f"{Fore.GREEN}• Loading central model weights...")
    weights_path = os.path.join(os.getcwd(), 'experiments', config['experiment_name'], 'model_weights', 'central_model_weights.npz')
    central_weights = load_model_weights(weights_path)
    
    local_training_history = {}
    print(f"{Fore.GREEN}• Beginning client training...")
    # Train each client
    for client in range(1, config['num_clients'] + 1):
        client_history, history_dict = train_client(client, round_num, central_weights, config, history_dict)
        local_training_history.update(client_history)
    
    print(f"{Fore.GREEN}• Aggregating client weights using FedAvg...")
    # Aggregate weights
    client_weights_path = os.path.join(os.getcwd(), 'experiments', config['experiment_name'], 'model_weights', 'client_weights')
    fed_avg_from_disk(client_weights_path, local_training_history)
    
    print(f"{Fore.GREEN}✓ Round {round_num} complete!")
    return history_dict

def save_round_history(round_num, history_dict, experiment_paths: ExperimentPaths):
    """Save history for a specific round"""
    round_dir = os.path.join(experiment_paths.history, f'round_{round_num}')
    create_directory_if_not_exists(round_dir)
    
    # Save round-specific history
    round_history_path = os.path.join(round_dir, 'history.pkl')
    round_data = {
        client: history_dict[client].get(f'round_{round_num}', {})
        for client in history_dict
        if client.startswith('client_')
    }
    
    with open(round_history_path, 'wb') as f:
        pickle.dump(round_data, f)

def save_history(history_dict, config, experiment_paths: ExperimentPaths):
    """Save complete history and round-specific histories"""
    # Save complete history
    history_path = os.path.join(experiment_paths.history, 'complete_history.pkl')
    create_directory_if_not_exists(os.path.dirname(history_path))   
    with open(history_path, 'wb') as f:
        pickle.dump(history_dict, f)
    
    # Save individual round histories
    for round_num in range(1, config['num_rounds'] + 1):
        save_round_history(round_num, history_dict, experiment_paths)

def run_federated_learning(config, experiment_paths: ExperimentPaths):
    start_time = datetime.now()
    log_section("Starting Federated Learning Process")
    print(f"{Fore.GREEN}Configuration:")
    print(f"{Fore.WHITE}• Experiment Name: {config['experiment_name']}")
    print(f"{Fore.WHITE}• Number of Clients: {config['num_clients']}")
    print(f"{Fore.WHITE}• Number of Rounds: {config['num_rounds']}")
    print(f"{Fore.WHITE}• Local Epochs: {config['local_epochs']}\n")
    
    # Initialize federated logger
    logger = FederatedLogger(config, experiment_paths.base)
    
    setup_gpu()
    # Initialize models and history
    history_dict = {}
    history_dict = initialize_central_model(config, experiment_paths, history_dict)
    history_dict = initialize_client_models(config, history_dict, experiment_paths)
    
    # Run federated learning rounds
    for round_num in range(1, config['num_rounds'] + 1):
        log_section(f"Training Round {round_num}/{config['num_rounds']}")
        
        print(f"{Fore.GREEN}• Loading central model weights...")
        weights_path = os.path.join(os.getcwd(), 'experiments', config['experiment_name'], 'weights', 'central_model_weights.npz')
        central_weights = load_model_weights(weights_path)
        
        # Load central model for logging
        central_model = load_model_from_disk(os.path.join(os.getcwd(), 'experiments', config['experiment_name'], 'models', 'central_model.keras'))
        central_model.set_weights(central_weights)
        
        local_training_history = {}
        client_histories = []
        metrics_list = []
        
        print(f"{Fore.GREEN}• Beginning client training...")
        # Train each client
        for client in range(1, config['num_clients'] + 1):
            client_history, history_dict = train_client(client, round_num, central_weights, config, history_dict, experiment_paths)
            local_training_history.update(client_history)
            
            # Load client model for logging, process it, and immediately clear it
            client_model = load_model_from_disk(os.path.join(os.getcwd(), 'experiments', config['experiment_name'], 'models', f'client_{client}.keras'))
            
            # Convert history to dict format
            history = history_dict[f'client_{client}'][f'round_{round_num}']
            history_dict = {k: v for k, v in history.history.items()}
            client_histories.append(history_dict)
            
            # Collect metrics
            metrics_list.append({
                'client_id': client,
                'round': round_num,
                'samples': client_history[f'client_{client}'],
                'loss': history_dict['loss'][-1],
                'accuracy': history_dict['accuracy'][-1]
            })
            
            # Log round information for this client immediately
            logger.log_round(round_num, central_model, [client_model], [history_dict], [metrics_list[-1]])
            
            # Clear client model from memory immediately
            clear_model_from_memory(client_model)
        
        print(f"{Fore.GREEN}• Aggregating client weights using FedAvg...")
        # Aggregate weights
        client_weights_path = os.path.join(os.getcwd(), 'experiments', config['experiment_name'], 'weights', 'client_weights')
        fed_avg_from_disk(client_weights_path, local_training_history)
        
        # Clear central model from memory
        clear_model_from_memory(central_model)
        
        # Save round history after each round
        save_round_history(round_num, history_dict, experiment_paths)
        
        print(f"{Fore.GREEN}✓ Round {round_num} complete!")
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    log_section("Federated Learning Complete")
    print(f"{Fore.GREEN}• Total Duration: {duration}")
    print(f"{Fore.GREEN}• Saving final history...")
    save_history(history_dict, config, experiment_paths)
    print(f"{Fore.GREEN}✓ Process completed successfully!")
    
    return history_dict

# def main():
#     config = {
#         'experiment_name': 'fl_template',
#         'num_clients': 5,
#         'num_rounds': 10,
#         'local_epochs': 1
#     }
#     run_federated_learning(config)

# if __name__ == "__main__":
#     main()