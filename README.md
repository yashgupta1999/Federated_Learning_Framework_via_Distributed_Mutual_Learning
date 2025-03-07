# Federated Learning for Face Mask Detection

This project implements federated learning approaches for face mask detection using TensorFlow and Keras.

## Project Overview

The project demonstrates two different federated learning approaches:

1. **Federated Averaging (FedAvg)** - A synchronous federated learning approach where client models are averaged after each round.
2. **Federated Distillation (FedKD)** - A knowledge distillation-based approach where client models learn from each other's predictions.

Both approaches are compared with a traditional centralized learning approach.

## Dataset

The code expects a dataset organized as follows:

```
├── Test/
│   ├── Mask/          # Training images with masks
│   └── NoMask/        # Training images without masks
└── Global/
    ├── Mask/          # Testing images with masks
    └── NoMask/        # Testing images without masks
```

## Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- PIL (Pillow)

Install all dependencies:

```bash
pip install -r requirements.txt
```

## Usage

The main script `fed_learning.py` supports three different training modes:

1. **Federated Averaging (FedAvg)**:
```bash
python fed_learning.py --mode fedavg
```

2. **Federated Distillation (FedKD)**:
```bash
python fed_learning.py --mode fedkd
```

3. **Centralized Training**:
```bash
python fed_learning.py --mode central
```

### Additional Options

- `--train-mask-dir` - Directory with mask images for training (default: `Test/Mask`)
- `--train-no-mask-dir` - Directory with no-mask images for training (default: `Test/NoMask`) 
- `--test-mask-dir` - Directory with mask images for testing (default: `Global/Mask`)
- `--test-no-mask-dir` - Directory with no-mask images for testing (default: `Global/NoMask`)
- `--num-clients` - Number of clients for federated learning (default: 5)
- `--num-rounds` - Number of communication rounds (default: 10)
- `--epochs` - Number of epochs per round (default: 10)
- `--img-size` - Image size for model input (default: 100)
- `--output-dir` - Directory to save results (default: Results)

## Architecture

The codebase follows an object-oriented approach with these main components:

- `DataManager` - Handles data loading, preprocessing, and splitting
- `ModelBuilder` - Creates and initializes CNN models
- `FederatedLearning` - Base class for federated learning implementations
- `FederatedAverage` - Implements the FedAvg algorithm
- `FederatedDistillation` - Implements the knowledge distillation-based approach
- `CentralizedTraining` - Implements traditional centralized training

## Results

Training results are saved in the `Results` directory and include:
- Test accuracy for each client
- Training loss history
- Training accuracy history

Visualizations of the training history are saved in the `Figs` directory.

TensorBoard logs are saved in the `logs/fit` directory and can be viewed with:

```bash
tensorboard --logdir logs/fit
```

## Citation

If you use this code for your research, please cite:

```
@misc{federated-mask-detection,
  author = {Your Name},
  title = {Federated Learning for Face Mask Detection},
  year = {2023},
  publisher = {GitHub},
  url = {https://github.com/yourusername/federated-mask-detection}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

# Run both approaches and compare results
python run_experiments.py --run-all

# Run just one approach
python run_experiments.py --run-fedavg

# Compare existing results
python run_experiments.py --compare

# Customize parameters
python run_experiments.py --run-all --num-clients 5 --num-rounds 10 --epochs 15 