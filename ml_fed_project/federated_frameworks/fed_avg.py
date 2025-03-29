import sys
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from data_utils import load_training_data
from model_utils import (
    create_model, 
    train_model, 
    save_model, 
    save_model_weights, 
    load_model_from_disk,
    load_model_weights,
    clear_model_from_memory
)

from fed_utils import fed_avg_from_disk
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

def initialize_central_model(config, history_dict=None):
    """Initialize and train the central model"""
    log_section("Initializing Central Model")
    print(f"{Fore.GREEN}• Creating central model architecture...")
    
    if history_dict is None:
        history_dict = {}
    
    model = create_model(config['seed'])
    data_path = os.path.join(os.getcwd(), 'experiments', config['experiment_name'], 'processed_data', 'init.npy')
    print(f"{Fore.GREEN}• Loading initial training data from: {data_path}")
    trainx, trainy = load_training_data(data_path)
    print(f"{Fore.GREEN}• Training central model with {len(trainx)} samples...")
    
    history = train_model(model, trainx, trainy)
    history_dict['init'] = history
    
    # Save central model
    model_path = os.path.join(os.getcwd(), 'experiments', config['experiment_name'], 'models', 'central_model.keras')
    weights_path = os.path.join(os.getcwd(), 'experiments', config['experiment_name'], 'model_weights')
    
    save_model(model, model_path)
    save_model_weights(model, 'central_model', weights_path)
    clear_model_from_memory(model)
    
    print(f"{Fore.GREEN}• Saving central model and weights...")
    
    print(f"{Fore.GREEN}✓ Central model initialization complete!")
    return history_dict

def initialize_client_models(config, history_dict=None):
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
        
        model_path = os.path.join(os.getcwd(), 'experiments', config['experiment_name'], 'models', f'{client_name}.keras')
        save_model(model, model_path)
        history_dict[client_name] = {}
        clear_model_from_memory(model)
    
    print(f"{Fore.GREEN}✓ All client models initialized successfully!")
    return history_dict

def train_client(client_num, round_num, central_weights, config, history_dict):
    """Train an individual client"""
    client_name = f'client_{client_num}'
    log_subsection(f"Training {client_name} - Round {round_num}")
    
    print(f"{Fore.WHITE}• Loading model and setting weights...")
    model_path = os.path.join(os.getcwd(), 'experiments', config['experiment_name'], 'models', f'client_{client_num}.keras')
    model = load_model_from_disk(model_path)
    model.set_weights(central_weights)
    
    print(f"{Fore.WHITE}• Loading training data...")
    data_path = os.path.join(os.getcwd(), 'experiments', config['experiment_name'], 'processed_data', f'client_{client_num}', f'round{round_num}.npy')
    trainx, trainy = load_training_data(data_path)
    print(f"{Fore.WHITE}• Training with {len(trainx)} samples for {config['local_epochs']} epochs...")
    
    history = train_model(model, trainx, trainy, epochs=config['local_epochs'])
    
    print(f"{Fore.WHITE}• Saving model and weights...")
    history_dict[client_name][f'round_{round_num}'] = history
    save_model(model, model_path)
    
    weights_path = os.path.join(os.getcwd(), 'experiments', config['experiment_name'], 'model_weights', 'client_weights')
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

def save_history(history_dict, config):
    history_path = os.path.join(os.getcwd(), 'experiments', config['experiment_name'], 'history', 'history.pkl')
    create_directory_if_not_exists(os.path.dirname(history_path))   
    with open(history_path, 'wb') as f:
        pickle.dump(history_dict, f)

def run_federated_learning(config):
    start_time = datetime.now()
    log_section("Starting Federated Learning Process")
    print(f"{Fore.GREEN}Configuration:")
    print(f"{Fore.WHITE}• Experiment Name: {config['experiment_name']}")
    print(f"{Fore.WHITE}• Number of Clients: {config['num_clients']}")
    print(f"{Fore.WHITE}• Number of Rounds: {config['num_rounds']}")
    print(f"{Fore.WHITE}• Local Epochs: {config['local_epochs']}\n")
    
    setup_gpu()
    # Initialize models and history
    history_dict = {}
    history_dict = initialize_central_model(config, history_dict)
    history_dict = initialize_client_models(config, history_dict)
    
    # Run federated learning rounds
    for round_num in range(1, config['num_rounds'] + 1):
        history_dict = train_round(round_num, config, history_dict)
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    log_section("Federated Learning Complete")
    print(f"{Fore.GREEN}• Total Duration: {duration}")
    print(f"{Fore.GREEN}• Saving final history...")
    save_history(history_dict, config)
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