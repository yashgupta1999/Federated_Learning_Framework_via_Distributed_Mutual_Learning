#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Federated Learning Implementation for Face Mask Detection

This module implements two approaches to federated learning:
1. Synchronous Federated Averaging (FedAvg)
2. Federated Learning with Knowledge Distillation (FedKD)

The code is organized into modular components for data loading, model creation,
training, and evaluation.
"""

import os
import random
import argparse
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import load_img
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Set random seeds for reproducibility
RANDOM_SEED = 333
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("federated_learning.log")
    ]
)
logger = logging.getLogger(__name__)


class DataManager:
    """
    Handles data loading, preprocessing, and splitting for federated learning.
    """
    
    def __init__(self, mask_dir, no_mask_dir, img_size=100):
        """
        Initialize the DataManager.
        
        Args:
            mask_dir (str): Directory containing mask images.
            no_mask_dir (str): Directory containing no mask images.
            img_size (int): Target image size for preprocessing.
        """
        self.mask_dir = mask_dir
        self.no_mask_dir = no_mask_dir
        self.img_size = img_size
        
    @staticmethod
    def get_image_files(directory):
        """Get all image files from a directory."""
        return [os.path.join(directory, f) for f in os.listdir(directory) 
                if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    
    def create_dataframe(self):
        """Create a dataframe with image paths and labels."""
        mask_images = self.get_image_files(self.mask_dir)
        mask_labels = np.zeros(len(mask_images))
        
        no_mask_images = self.get_image_files(self.no_mask_dir)
        no_mask_labels = np.ones(len(no_mask_images))
        
        # Create dataframes for each class
        mask_df = pd.DataFrame({"Image": mask_images, "Class": mask_labels})
        no_mask_df = pd.DataFrame({"Image": no_mask_images, "Class": no_mask_labels})
        
        # Concatenate and shuffle
        combined_df = pd.concat([mask_df, no_mask_df])
        return shuffle(combined_df, random_state=RANDOM_SEED)
    
    def create_federated_splits(self, df, num_clients, num_rounds, test_size=0.1):
        """
        Split data for federated learning.
        
        Args:
            df (DataFrame): DataFrame with image paths and labels.
            num_clients (int): Number of clients.
            num_rounds (int): Number of training rounds.
            test_size (float): Proportion of data to use for testing.
            
        Returns:
            tuple: (client_data, server_data, test_data)
        """
        # First separate out test data
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=RANDOM_SEED, stratify=df["Class"])
        
        # Split by class for stratified sampling
        mask_df = train_df[train_df["Class"] == 0]
        no_mask_df = train_df[train_df["Class"] == 1]
        
        # Calculate number of splits needed: (clients+1) per round + 1 for global init
        num_splits = (num_clients + 1) * num_rounds + 1
        
        # Split each class into equal folds
        mask_folds = self._create_folds(mask_df, num_splits)
        no_mask_folds = self._create_folds(no_mask_df, num_splits)
        
        # Merge corresponding folds from each class
        folds = []
        for i in range(num_splits):
            fold = pd.concat([mask_folds[i], no_mask_folds[i]])
            folds.append(shuffle(fold, random_state=RANDOM_SEED))
        
        return folds, test_df
    
    def _create_folds(self, df, num_splits):
        """Split dataframe into stratified folds."""
        indices = np.array_split(shuffle(df.index, random_state=RANDOM_SEED), num_splits)
        return [df.loc[idx] for idx in indices]
    
    def load_and_preprocess_data(self, df):
        """
        Load and preprocess images from dataframe.
        
        Args:
            df (DataFrame): DataFrame containing image paths and labels.
            
        Returns:
            tuple: (X, y) - preprocessed images and labels
        """
        data = []
        
        for img_path in df["Image"]:
            image = load_img(img_path, target_size=(self.img_size, self.img_size))
            image = img_to_array(image)
            image = image / 255.0  # Normalize to [0,1]
            data.append(image)
            
        X = np.array(data, dtype="float32")
        y = np.array(df["Class"]).reshape([-1, 1])
        
        return X, y


class ModelBuilder:
    """
    Creates and initializes models for federated learning.
    """
    
    @staticmethod
    def create_mask_detection_model(input_shape=(100, 100, 3)):
        """
        Create a CNN model for mask detection.
        
        Args:
            input_shape (tuple): Input shape of images.
            
        Returns:
            Model: Compiled TensorFlow model.
        """
        model = Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=[
                tf.keras.metrics.Recall(),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.AUC(
                    num_thresholds=200,
                    curve='ROC',
                    summation_method='interpolation'
                ),
                'accuracy'
            ]
        )
        
        return model


class KLLoss(tf.keras.losses.Loss):
    """
    Kullback-Leibler divergence loss for knowledge distillation.
    """
    
    def __init__(self):
        super().__init__()
        self.kl = tf.keras.losses.KLDivergence()
        
    def call(self, y_true, y_pred_list):
        """
        Calculate KL divergence loss between true values and predictions.
        
        Args:
            y_true: True labels
            y_pred_list: List of model predictions
            
        Returns:
            float: Mean KL divergence
        """
        kl_values = []
        
        for pred in y_pred_list:
            kl_values.append(self.kl(y_true, pred))
            
        return tf.reduce_mean(tf.convert_to_tensor(kl_values, dtype=tf.float32))


class FederatedLearning:
    """Base class for federated learning implementations."""
    
    def __init__(self, num_clients, num_rounds, data_manager, model_builder):
        """
        Initialize federated learning.
        
        Args:
            num_clients (int): Number of clients.
            num_rounds (int): Number of training rounds.
            data_manager (DataManager): Data manager instance.
            model_builder (ModelBuilder): Model builder instance.
        """
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.data_manager = data_manager
        self.model_builder = model_builder
        
        # Initialize client models and metrics
        self.client_models = []
        self.global_model = None
        self.client_acc_history = {}
        self.client_loss_history = {}
        
        # Setup TensorBoard logging
        self.log_dir = f"logs/fit/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir)
    
    def initialize_models(self):
        """Initialize global and client models."""
        self.global_model = self.model_builder.create_mask_detection_model()
        
        # Create client models by copying global model weights
        self.client_models = []
        for i in range(self.num_clients):
            model = self.model_builder.create_mask_detection_model()
            model.set_weights(self.global_model.get_weights())
            model._name = f"Client_{i}"
            self.client_models.append(model)
            
            # Initialize history tracking
            self.client_acc_history[i] = []
            self.client_loss_history[i] = []
    
    def train_model(self, model, data_fold, epochs=10, verbose=0):
        """
        Train a model on a data fold.
        
        Args:
            model (Model): TensorFlow model to train.
            data_fold (DataFrame): Data fold to train on.
            epochs (int): Number of training epochs.
            verbose (int): Verbosity level.
            
        Returns:
            tuple: (trained_model, accuracy_history, loss_history)
        """
        X, y = self.data_manager.load_and_preprocess_data(data_fold)
        
        history = model.fit(
            X, y,
            epochs=epochs,
            validation_split=0.2,
            verbose=verbose,
            callbacks=[self.tensorboard_callback]
        )
        
        accuracy = history.history['accuracy']
        loss = history.history['loss']
        
        return model, accuracy, loss
    
    def evaluate_models(self, test_data):
        """
        Evaluate all client models on test data.
        
        Args:
            test_data (DataFrame): Test data.
            
        Returns:
            list: Test accuracies for each client.
        """
        X_test, y_test = self.data_manager.load_and_preprocess_data(test_data)
        
        results = []
        for client_idx, model in enumerate(self.client_models):
            evaluation = model.evaluate(X_test, y_test, verbose=1)
            logger.info(f"Client {client_idx} test accuracy: {evaluation[-1]:.4f}")
            results.append(evaluation[-1])  # Accuracy is the last metric
            
        return results
    
    def save_results(self, results, output_dir="Results"):
        """
        Save evaluation results and training history.
        
        Args:
            results (list): Evaluation results.
            output_dir (str): Output directory.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save test results
        df_results = pd.DataFrame({"Testing_Results": results})
        df_results.to_csv(f"{output_dir}/{self.__class__.__name__}_results.csv")
        
        # Save loss history
        df_loss = pd.DataFrame()
        for k in self.client_loss_history.keys():
            df_loss[f"Client_{k}"] = self.client_loss_history[k]
        df_loss.to_csv(f"{output_dir}/{self.__class__.__name__}_loss.csv", index=False)
        
        # Save accuracy history
        df_acc = pd.DataFrame()
        for k in self.client_acc_history.keys():
            df_acc[f"Client_{k}"] = self.client_acc_history[k]
        df_acc.to_csv(f"{output_dir}/{self.__class__.__name__}_accuracy.csv", index=False)
        
        logger.info(f"Results saved to {output_dir}")
    
    def plot_history(self, client_idx=None, metric='accuracy'):
        """
        Plot training history for specific client or all clients.
        
        Args:
            client_idx (int, optional): Client index or None for all clients.
            metric (str): Metric to plot ('accuracy' or 'loss').
        """
        plt.figure(figsize=(10, 6))
        
        if client_idx is not None:
            # Plot single client
            history = self.client_acc_history[client_idx] if metric == 'accuracy' else self.client_loss_history[client_idx]
            plt.plot(history, label=f'Client {client_idx}')
        else:
            # Plot all clients
            for idx in range(self.num_clients):
                history = self.client_acc_history[idx] if metric == 'accuracy' else self.client_loss_history[idx]
                plt.plot(history, label=f'Client {idx}')
        
        plt.title(f'{metric.capitalize()} History')
        plt.xlabel('Training Step')
        plt.ylabel(metric.capitalize())
        plt.legend()
        
        # Save the plot
        os.makedirs("Figs", exist_ok=True)
        plt.savefig(f"Figs/{self.__class__.__name__}_{metric}.png")
        plt.close()


class FederatedAverage(FederatedLearning):
    """Implementation of Federated Averaging (FedAvg) algorithm."""
    
    def train(self, data_folds, test_data, epochs=10):
        """
        Train using Federated Averaging.
        
        Args:
            data_folds (list): List of data folds.
            test_data (DataFrame): Test data.
            epochs (int): Number of epochs per round.
            
        Returns:
            list: Final test accuracies.
        """
        # Initialize models
        self.initialize_models()
        
        # Initial global model training
        global_fold = data_folds.pop(0)
        self.global_model, global_acc, global_loss = self.train_model(
            self.global_model, global_fold, epochs=epochs)
        
        # Initialize history with global model results
        for client_idx in range(self.num_clients):
            self.client_acc_history[client_idx] = global_acc.copy()
            self.client_loss_history[client_idx] = global_loss.copy()
        
        # For each round
        for round_idx in range(self.num_rounds):
            logger.info(f"Round {round_idx + 1}/{self.num_rounds}")
            
            # Train each client
            for client_idx in range(self.num_clients):
                client_fold = data_folds.pop(0)
                self.client_models[client_idx], acc, loss = self.train_model(
                    self.client_models[client_idx], client_fold, epochs=epochs)
                
                # Update history
                self.client_acc_history[client_idx].extend(acc)
                self.client_loss_history[client_idx].extend(loss)
            
            # Perform federated averaging
            self._federated_averaging()
        
        # Final evaluation
        return self.evaluate_models(test_data)
    
    def _federated_averaging(self):
        """Perform federated averaging of model weights."""
        # Collect client weights
        weights = [model.get_weights() for model in self.client_models]
        
        # Average weights
        avg_weights = self._average_weights(weights)
        
        # Update all client models with the averaged weights
        for model in self.client_models:
            model.set_weights(avg_weights)
    
    @staticmethod
    def _average_weights(weights_list):
        """
        Average model weights.
        
        Args:
            weights_list (list): List of model weights.
            
        Returns:
            list: Averaged weights.
        """
        avg_weights = [np.zeros_like(w) for w in weights_list[0]]
        
        for weights in weights_list:
            for i, w in enumerate(weights):
                avg_weights[i] += w
        
        for i in range(len(avg_weights)):
            avg_weights[i] /= len(weights_list)
            
        return avg_weights


class FederatedDistillation(FederatedLearning):
    """Implementation of Federated Learning with Knowledge Distillation."""
    
    def train(self, data_folds, test_data, epochs=10):
        """
        Train using Federated Distillation.
        
        Args:
            data_folds (list): List of data folds.
            test_data (DataFrame): Test data.
            epochs (int): Number of epochs per round.
            
        Returns:
            list: Final test accuracies.
        """
        # Initialize models
        self.initialize_models()
        
        # Initial global model training
        global_fold = data_folds.pop(0)
        self.global_model, global_acc, global_loss = self.train_model(
            self.global_model, global_fold, epochs=epochs)
        
        # Initialize history with global model results
        for client_idx in range(self.num_clients):
            self.client_acc_history[client_idx] = global_acc.copy()
            self.client_loss_history[client_idx] = global_loss.copy()
        
        # For each round
        for round_idx in range(self.num_rounds):
            logger.info(f"Round {round_idx + 1}/{self.num_rounds}")
            
            # Train each client
            for client_idx in range(self.num_clients):
                client_fold = data_folds.pop(0)
                self.client_models[client_idx], acc, loss = self.train_model(
                    self.client_models[client_idx], client_fold, epochs=epochs)
                
                # Update history
                self.client_acc_history[client_idx].extend(acc)
                self.client_loss_history[client_idx].extend(loss)
            
            # Perform knowledge distillation
            distillation_fold = data_folds.pop(0)
            self._knowledge_distillation(distillation_fold)
        
        # Final evaluation
        return self.evaluate_models(test_data)
    
    def _knowledge_distillation(self, data_fold):
        """
        Perform knowledge distillation among clients.
        
        Args:
            data_fold (DataFrame): Data fold for distillation.
        """
        X, y = self.data_manager.load_and_preprocess_data(data_fold)
        
        # Get predictions from all clients
        client_predictions = []
        for client_idx, model in enumerate(self.client_models):
            preds = model.predict(X)
            client_predictions.append(preds)
            
            # Get evaluation metrics
            eval_metrics = model.evaluate(X, y, verbose=0)
            self.client_loss_history[client_idx].append(eval_metrics[0])
            self.client_acc_history[client_idx].append(eval_metrics[-1])
        
        # For each client, perform distillation using other clients' predictions
        for client_idx in range(self.num_clients):
            # Get predictions from all other clients
            other_predictions = [pred for i, pred in enumerate(client_predictions) 
                               if i != client_idx]
            
            # Fine-tune model using KL divergence
            updated_model, acc_hist, loss_hist = self._optimize_weights(
                self.client_models[client_idx], X, y, other_predictions)
            
            # Update model and history
            self.client_models[client_idx] = updated_model
            self.client_acc_history[client_idx].extend(acc_hist)
            self.client_loss_history[client_idx].extend(loss_hist)
    
    def _optimize_weights(self, model, X, y, other_predictions, epochs=10, batch_size=32):
        """
        Optimize model using knowledge distillation.
        
        Args:
            model (Model): Model to optimize.
            X (ndarray): Input data.
            y (ndarray): True labels.
            other_predictions (list): Predictions from other models.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size.
            
        Returns:
            tuple: (optimized_model, accuracy_history, loss_history)
        """
        optimizer = tf.keras.optimizers.Adam()
        binary_crossentropy = tf.keras.losses.BinaryCrossentropy()
        kl_loss = KLLoss()
        
        loss_history = []
        accuracy_history = []
        
        # Create batches
        num_samples = len(X)
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        X_batches = np.array_split(X, num_batches)
        y_batches = np.array_split(y, num_batches)
        
        # Split predictions from other models into batches
        other_pred_batches = []
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            batch_preds = [pred[start_idx:end_idx] for pred in other_predictions]
            other_pred_batches.append(batch_preds)
        
        # Training loop
        for epoch in range(epochs):
            epoch_loss_avg = tf.keras.metrics.Mean()
            epoch_accuracy = tf.keras.metrics.BinaryAccuracy()
            
            # Batch training
            for x_batch, y_batch, pred_batch in zip(X_batches, y_batches, other_pred_batches):
                with tf.GradientTape() as tape:
                    # Forward pass
                    y_pred = model(x_batch, training=True)
                    
                    # Combined loss: CE + KL
                    ce_loss = binary_crossentropy(y_batch, y_pred)
                    distillation_loss = kl_loss(y_pred, pred_batch)
                    total_loss = ce_loss + distillation_loss
                
                # Compute gradients and apply updates
                gradients = tape.gradient(total_loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                
                # Update metrics
                epoch_loss_avg.update_state(total_loss)
                epoch_accuracy.update_state(y_batch, y_pred)
            
            # Track progress
            loss_history.append(epoch_loss_avg.result().numpy())
            accuracy_history.append(epoch_accuracy.result().numpy())
            
        return model, accuracy_history, loss_history


class CentralizedTraining:
    """Implementation of centralized training (for comparison)."""
    
    def __init__(self, data_manager, model_builder):
        """
        Initialize centralized training.
        
        Args:
            data_manager (DataManager): Data manager instance.
            model_builder (ModelBuilder): Model builder instance.
        """
        self.data_manager = data_manager
        self.model_builder = model_builder
        self.model = None
        
    def train(self, data, test_data, epochs=10):
        """
        Train a centralized model.
        
        Args:
            data (DataFrame): Training data.
            test_data (DataFrame): Test data.
            epochs (int): Number of training epochs.
            
        Returns:
            tuple: (model, test_accuracy)
        """
        self.model = self.model_builder.create_mask_detection_model()
        
        X, y = self.data_manager.load_and_preprocess_data(data)
        X_test, y_test = self.data_manager.load_and_preprocess_data(test_data)
        
        # Train the model
        history = self.model.fit(
            X, y,
            epochs=epochs,
            validation_split=0.2,
            verbose=1
        )
        
        # Evaluate the model
        evaluation = self.model.evaluate(X_test, y_test, verbose=1)
        test_accuracy = evaluation[-1]
        
        logger.info(f"Centralized model test accuracy: {test_accuracy:.4f}")
        
        return self.model, test_accuracy
    
    def save_model(self, filename="centralized_model.h5"):
        """Save the trained model."""
        if self.model is not None:
            self.model.save(filename)
            logger.info(f"Model saved to {filename}")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Federated Learning for Mask Detection")
    
    parser.add_argument("--mode", type=str, default="fedavg",
                        choices=["fedavg", "fedkd", "central"],
                        help="Training mode: fedavg (Federated Averaging), fedkd (Federated Distillation), central (Centralized)")
    
    parser.add_argument("--train-mask-dir", type=str, default="Test/Mask",
                        help="Directory containing mask images for training")
    
    parser.add_argument("--train-no-mask-dir", type=str, default="Test/NoMask",
                        help="Directory containing no-mask images for training")
    
    parser.add_argument("--test-mask-dir", type=str, default="Global/Mask",
                        help="Directory containing mask images for testing")
    
    parser.add_argument("--test-no-mask-dir", type=str, default="Global/NoMask",
                        help="Directory containing no-mask images for testing")
    
    parser.add_argument("--num-clients", type=int, default=5,
                        help="Number of clients for federated learning")
    
    parser.add_argument("--num-rounds", type=int, default=10,
                        help="Number of communication rounds")
    
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs per round")
    
    parser.add_argument("--img-size", type=int, default=100,
                        help="Image size for model input")
    
    parser.add_argument("--output-dir", type=str, default="Results",
                        help="Directory to save results")
    
    return parser.parse_args()


def main():
    """Main function to run the experiment."""
    args = parse_arguments()
    
    # Set up data manager
    data_manager = DataManager(
        mask_dir=args.train_mask_dir,
        no_mask_dir=args.train_no_mask_dir,
        img_size=args.img_size
    )
    
    # Create model builder
    model_builder = ModelBuilder()
    
    # Load data
    logger.info("Loading training data...")
    train_df = data_manager.create_dataframe()
    
    # Load test data
    logger.info("Loading test data...")
    test_data_manager = DataManager(
        mask_dir=args.test_mask_dir,
        no_mask_dir=args.test_no_mask_dir,
        img_size=args.img_size
    )
    test_df = test_data_manager.create_dataframe()
    
    # Run selected mode
    if args.mode == "central":
        logger.info("Running centralized training...")
        central_trainer = CentralizedTraining(data_manager, model_builder)
        model, accuracy = central_trainer.train(train_df, test_df, epochs=args.epochs)
        central_trainer.save_model()
        
    else:
        # Create data splits for federated learning
        logger.info("Creating federated data splits...")
        data_folds, _ = data_manager.create_federated_splits(
            train_df, args.num_clients, args.num_rounds)
        
        if args.mode == "fedavg":
            logger.info("Running Federated Averaging...")
            fed_avg = FederatedAverage(
                args.num_clients, args.num_rounds, data_manager, model_builder)
            results = fed_avg.train(data_folds, test_df, epochs=args.epochs)
            fed_avg.save_results(results, args.output_dir)
            fed_avg.plot_history(metric='accuracy')
            fed_avg.plot_history(metric='loss')
            
        elif args.mode == "fedkd":
            logger.info("Running Federated Distillation...")
            fed_kd = FederatedDistillation(
                args.num_clients, args.num_rounds, data_manager, model_builder)
            results = fed_kd.train(data_folds, test_df, epochs=args.epochs)
            fed_kd.save_results(results, args.output_dir)
            fed_kd.plot_history(metric='accuracy')
            fed_kd.plot_history(metric='loss')
    
    logger.info("Experiment completed successfully!")


if __name__ == "__main__":
    main() 