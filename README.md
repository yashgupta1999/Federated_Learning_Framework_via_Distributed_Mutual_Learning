# Federated Learning Framework

A comprehensive framework for implementing and experimenting with federated learning algorithms. This project provides a flexible and extensible platform for training machine learning models in a distributed manner while keeping data private and local to each client.

## Project Structure

```
ml_fed_project/
├── config/                     # Configuration files
│   └── fl_template_config.yaml # Template configuration for experiments
├── config_utils/              # Configuration management utilities
├── data/                      # Dataset storage
├── data_utils/               # Data processing and management utilities
├── docs/                     # Documentation
├── experiment_utils/         # Experiment management utilities
├── federated_frameworks/     # Different federated learning implementations
│   ├── fed_avg.py           # Federated Averaging implementation
│   ├── fed_mutual_learning.py
│   ├── fed_conditioned_asynchronous.py
│   └── fed_conditioned_synchronous.py
├── fed_utils/               # Shared utilities for federated learning
├── main.py                  # Main entry point for running experiments
├── model_utils/             # Model creation and management utilities
└── notebooks/               # Jupyter notebooks for analysis
```

## Features

- **Multiple Federated Learning Algorithms**:
  - Federated Averaging (FedAvg)
  - Mutual Learning
  - Conditioned Asynchronous Learning
  - Conditioned Synchronous Learning

- **Flexible Configuration**:
  - YAML-based configuration system
  - Configurable number of clients
  - Adjustable training rounds and local epochs
  - Support for stratified data splitting

- **Data Management**:
  - Support for various datasets
  - Automatic data splitting and distribution
  - Stratified sampling options
  - Global model training option

- **Experiment Management**:
  - Organized experiment directory structure
  - Comprehensive logging and metrics tracking
  - Model checkpointing and weight management
  - GPU support

## Configuration

The framework uses a YAML-based configuration system. Here's an example configuration:

```yaml
experiment_name: "fed_test_experiment_1"
data_path: "./data/facemask_dataset"
num_clients: 2
num_rounds: 1
local_epochs: 1
seed: 333
include_global: true
stratified: true
```

## Usage

1. **Setup**:
   ```bash
   # Clone the repository
   git clone [repository-url]
   cd ml_fed_project
   
   # Install dependencies (create requirements.txt if not exists)
   pip install -r requirements.txt
   ```

2. **Configure Your Experiment**:
   - Copy `config/fl_template_config.yaml` to create your experiment configuration
   - Modify the parameters according to your needs

3. **Run the Experiment**:
   ```bash
   python main.py
   ```

## Project Components

### Federated Learning Frameworks
- `fed_avg.py`: Implements the standard Federated Averaging algorithm
- `fed_mutual_learning.py`: Implements mutual learning between clients
- `fed_conditioned_asynchronous.py`: Asynchronous federated learning with conditions
- `fed_conditioned_synchronous.py`: Synchronous federated learning with conditions

### Utilities
- `config_utils/`: Handles configuration loading and management
- `data_utils/`: Manages data processing, splitting, and distribution
- `model_utils/`: Handles model creation, training, and weight management
- `experiment_utils/`: Manages experiment setup, logging, and execution
- `fed_utils/`: Shared utilities for federated learning implementations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your license information here]

## Contact

[Add your contact information here] 