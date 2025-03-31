# Federated Learning Framework

A flexible and extensible framework for implementing and experimenting with federated learning algorithms, with a focus on computer vision tasks. This framework enables distributed training of machine learning models across multiple clients while maintaining data privacy and locality.

## Project Overview

This project implements a federated learning framework that allows for distributed training of machine learning models across multiple clients while keeping data local. The framework is particularly well-suited for computer vision tasks and includes support for various federated learning algorithms, with Federated Averaging (FedAvg) as the primary implementation.

### Key Benefits

- **Privacy-Preserving**: Data remains on local devices, never leaving the client
- **Distributed Training**: Efficient training across multiple clients
- **Flexible Architecture**: Easy to extend with new federated learning algorithms
- **Comprehensive Monitoring**: Detailed tracking of training progress and metrics
- **Resource Efficient**: Built-in GPU memory management and optimization

## Project Structure

```
ml_fed_project/
├── config/                 # Configuration files and templates
│   └── fl_template_config.yaml  # Template for experiment configuration
├── config_utils/          # Configuration management utilities
│   └── paths.py          # Path management for experiments
├── data/                  # Dataset storage and management
├── data_utils/           # Data processing and management utilities
│   ├── data_processing.py    # Data preprocessing functions
│   └── data_splitting.py     # Data distribution utilities
├── docs/                 # Project documentation
├── experiment_utils/     # Experiment management utilities
│   ├── directory.py      # Directory management
│   └── gpu.py           # GPU configuration
├── fed_utils/           # Federated learning specific utilities
│   └── fed_avg.py       # FedAvg implementation
├── federated_frameworks/ # Core federated learning implementations
├── model_utils/         # Model creation and management utilities
├── notebooks/           # Jupyter notebooks for analysis
└── main.py             # Main entry point for running experiments
```

## Features

### Core Functionality

- **Federated Averaging (FedAvg) Implementation**
  - Distributed model training across multiple clients
  - Weight aggregation and synchronization
  - Support for heterogeneous client participation

- **Data Management**
  - Stratified data splitting for balanced distribution
  - Support for various data formats
  - Efficient data preprocessing pipeline
  - Local data storage and management

- **Model Management**
  - Flexible model architecture support
  - Weight saving and loading
  - Model checkpointing
  - Memory-efficient training

- **Experiment Management**
  - YAML-based configuration system
  - Comprehensive logging system
  - Metrics tracking and visualization
  - Experiment reproducibility

### Technical Features

- **GPU Support**
  - Automatic GPU detection and configuration
  - Memory growth management
  - Multi-GPU support

- **Performance Optimization**
  - Efficient weight aggregation
  - Memory management
  - Batch processing support

## Getting Started

### Prerequisites

- Python 3.7+
- TensorFlow 2.x
- NumPy
- PyYAML
- Other dependencies (see requirements.txt)

### Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd ml_fed_project
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running Experiments

1. Configure your experiment:
   - Copy `config/fl_template_config.yaml` to create your experiment config
   - Modify parameters as needed:
     ```yaml
     experiment_name: "your_experiment"
     data_path: "./data/your_dataset"
     num_clients: 5
     num_rounds: 10
     local_epochs: 5
     ```

2. Run the main script:
```bash
python main.py
```

3. Monitor progress:
   - Check the experiment directory for logs
   - View metrics in the generated CSV files
   - Analyze results using provided notebooks

## Configuration

### Key Configuration Parameters

- **Experiment Settings**
  - `experiment_name`: Unique identifier for the experiment
  - `base_dir`: Base directory for experiment outputs
  - `seed`: Random seed for reproducibility

- **Data Settings**
  - `data_path`: Path to the dataset
  - `num_clients`: Number of federated learning clients
  - `stratified`: Enable stratified sampling
  - `include_global`: Include global model training

- **Training Settings**
  - `num_rounds`: Number of federated learning rounds
  - `local_epochs`: Local training epochs per round
  - `batch_size`: Batch size for training
  - `learning_rate`: Learning rate for optimization

- **Logging Settings**
  - `directory_structure`: Customize output directory structure
  - `file_formats`: Configure file formats for weights and history

## Experiment Structure

### 1. Data Preparation
- Dataset loading and validation
- Data splitting across clients
- Preprocessing and augmentation
- Local data storage setup

### 2. Model Initialization
- Model architecture creation
- Weight initialization
- Client model setup
- Global model initialization

### 3. Federated Learning Process
- **Round Execution**
  - Local training on each client
  - Model weight collection
  - Weight aggregation
  - Global model update
  - Performance evaluation

- **Monitoring**
  - Training metrics tracking
  - Validation performance
  - Resource utilization
  - Progress logging

### 4. Results and Analysis
- Model checkpointing
- Performance metrics export
- Training history saving
- Results visualization

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add docstrings to new functions
- Include unit tests for new features
- Update documentation as needed

## License

[Add your license information here]

## Support

For issues and feature requests, please use the GitHub issue tracker.

## Acknowledgments

- TensorFlow Federated for inspiration
- Contributors and maintainers
- Open source community 