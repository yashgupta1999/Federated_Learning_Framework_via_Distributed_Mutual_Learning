#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to run and compare different federated learning approaches.
"""

import os
import argparse
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import seaborn as sns

def run_experiment(mode, num_clients=5, num_rounds=10, epochs=10):
    """Run a federated learning experiment."""
    print(f"\n{'='*80}\nRunning {mode.upper()} experiment\n{'='*80}")
    
    cmd = [
        "python", "fed_learning.py",
        "--mode", mode,
        "--num-clients", str(num_clients),
        "--num-rounds", str(num_rounds),
        "--epochs", str(epochs)
    ]
    
    subprocess.run(cmd, check=True)

def compare_results(output_dir="Results"):
    """Compare results from different experiments."""
    # Find result files
    fedavg_results = os.path.join(output_dir, "FederatedAverage_results.csv")
    fedkd_results = os.path.join(output_dir, "FederatedDistillation_results.csv")
    
    # Check if result files exist
    if not (os.path.exists(fedavg_results) and os.path.exists(fedkd_results)):
        print("Result files not found. Run both experiments first.")
        return
    
    # Load results
    fedavg_df = pd.read_csv(fedavg_results)
    fedkd_df = pd.read_csv(fedkd_results)
    
    # Calculate statistics
    fedavg_mean = fedavg_df["Testing_Results"].mean()
    fedavg_std = fedavg_df["Testing_Results"].std()
    
    fedkd_mean = fedkd_df["Testing_Results"].mean()
    fedkd_std = fedkd_df["Testing_Results"].std()
    
    # Print statistics
    print("\n" + "="*50)
    print("Results Comparison")
    print("="*50)
    print(f"FedAvg - Mean Accuracy: {fedavg_mean:.4f} ± {fedavg_std:.4f}")
    print(f"FedKD  - Mean Accuracy: {fedkd_mean:.4f} ± {fedkd_std:.4f}")
    
    # Create comparison chart
    plt.figure(figsize=(10, 6))
    
    # Prepare data for plotting
    methods = ["FedAvg", "FedKD"]
    means = [fedavg_mean, fedkd_mean]
    stds = [fedavg_std, fedkd_std]
    
    # Create bar chart
    x_pos = np.arange(len(methods))
    plt.bar(x_pos, means, yerr=stds, align='center', alpha=0.7, capsize=10)
    plt.xticks(x_pos, methods)
    plt.ylabel('Test Accuracy')
    plt.title('Comparison of Federated Learning Methods')
    
    # Add accuracy values on top of bars
    for i, v in enumerate(means):
        plt.text(i, v + stds[i] + 0.01, f"{v:.4f}", ha='center', fontweight='bold')
    
    # Save figure
    os.makedirs("Figs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    plt.savefig(f"Figs/comparison_{timestamp}.png")
    plt.tight_layout()
    plt.show()
    
    # Compare learning curves
    compare_learning_curves(output_dir)

def compare_learning_curves(output_dir="Results"):
    """Compare learning curves from different experiments."""
    # Find accuracy history files
    fedavg_acc = os.path.join(output_dir, "FederatedAverage_accuracy.csv")
    fedkd_acc = os.path.join(output_dir, "FederatedDistillation_accuracy.csv")
    
    fedavg_loss = os.path.join(output_dir, "FederatedAverage_loss.csv")
    fedkd_loss = os.path.join(output_dir, "FederatedDistillation_loss.csv")
    
    # Check if files exist
    if not all(os.path.exists(f) for f in [fedavg_acc, fedkd_acc, fedavg_loss, fedkd_loss]):
        print("History files not found. Run both experiments first.")
        return
    
    # Load data
    fedavg_acc_df = pd.read_csv(fedavg_acc)
    fedkd_acc_df = pd.read_csv(fedkd_acc)
    
    fedavg_loss_df = pd.read_csv(fedavg_loss)
    fedkd_loss_df = pd.read_csv(fedkd_loss)
    
    # Create plots
    fig, axes = plt.subplots(2, 1, figsize=(12, 14))
    
    # Plot accuracy
    ax1 = axes[0]
    # Calculate average accuracy across clients
    fedavg_mean_acc = fedavg_acc_df.mean(axis=1)
    fedkd_mean_acc = fedkd_acc_df.mean(axis=1)
    
    ax1.plot(fedavg_mean_acc, label='FedAvg', linewidth=2)
    ax1.plot(fedkd_mean_acc, label='FedKD', linewidth=2)
    ax1.set_title('Average Accuracy During Training')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot loss
    ax2 = axes[1]
    # Calculate average loss across clients
    fedavg_mean_loss = fedavg_loss_df.mean(axis=1)
    fedkd_mean_loss = fedkd_loss_df.mean(axis=1)
    
    ax2.plot(fedavg_mean_loss, label='FedAvg', linewidth=2)
    ax2.plot(fedkd_mean_loss, label='FedKD', linewidth=2)
    ax2.set_title('Average Loss During Training')
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs("Figs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    plt.savefig(f"Figs/learning_curves_{timestamp}.png")
    plt.show()

def analyze_client_performance(output_dir="Results"):
    """Analyze individual client performance similar to Result Analysis.ipynb."""
    print(f"\n{'='*80}\nAnalyzing client performance\n{'='*80}")
    
    # Create Figs directory if it doesn't exist
    os.makedirs("Figs", exist_ok=True)
    
    # Check for result files from different methods
    methods = {
        "FederatedAverage": "FedAvg",
        "FederatedDistillation": "FedKD"
    }
    
    # Dictionary to store dataframes
    dfs = {}
    
    # Try to load all available result files
    for method, short_name in methods.items():
        result_file = os.path.join(output_dir, f"{method}_results.csv")
        if os.path.exists(result_file):
            dfs[short_name] = pd.read_csv(result_file)
            print(f"Loaded results for {short_name}")
    
    if not dfs:
        print("No result files found. Run experiments first.")
        return
    
    # Create comparison dataframe for testing results
    comparison_df = pd.DataFrame()
    
    for method_name, df in dfs.items():
        if "Testing_Results" in df.columns:
            comparison_df[method_name] = df["Testing_Results"]
    
    # Save the comparison results
    comparison_df.to_csv(os.path.join(output_dir, "method_comparison.csv"), index=False)
    print(f"Saved method comparison to {os.path.join(output_dir, 'method_comparison.csv')}")
    
    # Generate bar plot comparing methods
    plt.figure(figsize=(10, 6))
    comparison_df.mean().plot(kind='bar', yerr=comparison_df.std(), capsize=10, rot=0)
    plt.title('Performance Comparison on Testing Dataset')
    plt.ylabel('Accuracy')
    plt.xlabel('Method')
    plt.tight_layout()
    plt.savefig(os.path.join("Figs", "method_comparison_bar.png"))
    plt.show()
    
    # Analyze loss curves for each method
    analyze_loss_curves(output_dir)

def analyze_loss_curves(output_dir="Results"):
    """Analyze and visualize loss curves with highlights for communication rounds."""
    # Try to load loss history for different methods
    for method in ["FederatedAverage", "FederatedDistillation"]:
        loss_file = os.path.join(output_dir, f"{method}_loss.csv")
        if not os.path.exists(loss_file):
            continue
            
        # Load and process loss data
        loss_df = pd.read_csv(loss_file)
        
        # Apply exponential weighted mean for smoothing (as in the notebook)
        loss_df = loss_df.ewm(alpha=0.1).mean()
        
        # Create plot
        plt.figure(figsize=(20, 5))
        
        # Plot client loss curves
        client_columns = [col for col in loss_df.columns if col.startswith("Client")]
        loss_df.plot(kind="line", y=client_columns, colormap="rocket_r", ax=plt.gca(), 
                    title=f"Training History for {method}")
        
        # Add vertical spans to highlight communication rounds
        # Assuming communication happens every 10 epochs as in the notebook
        for i in range(12):  # Adjust based on your experiment setup
            plt.axvspan(19+(i*10), 29+(i*10), alpha=0.1, color='black')
        
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(title="Clients")
        
        # Save the figure
        plt.savefig(os.path.join("Figs", f"{method}_loss.png"))
        plt.show()
        
        print(f"Generated loss curve visualization for {method}")

def comprehensive_analysis(output_dir="Results"):
    """Run a comprehensive analysis on all available data."""
    print(f"\n{'='*80}\nRunning comprehensive analysis\n{'='*80}")
    
    # Compare results if available
    try:
        compare_results(output_dir)
    except Exception as e:
        print(f"Error comparing results: {e}")
    
    # Analyze client performance
    try:
        analyze_client_performance(output_dir)
    except Exception as e:
        print(f"Error analyzing client performance: {e}")

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run and compare federated learning experiments.")
    
    parser.add_argument("--run-fedavg", action="store_true",
                        help="Run Federated Averaging experiment")
    
    parser.add_argument("--run-fedkd", action="store_true", 
                        help="Run Federated Distillation experiment")
    
    parser.add_argument("--run-all", action="store_true",
                        help="Run both experiments")
    
    parser.add_argument("--compare", action="store_true",
                        help="Compare results after running experiments")
    
    parser.add_argument("--analyze", action="store_true",
                        help="Run comprehensive analysis on experiment results")
    
    parser.add_argument("--num-clients", type=int, default=5,
                        help="Number of clients for federated learning")
    
    parser.add_argument("--num-rounds", type=int, default=10,
                        help="Number of communication rounds")
    
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs per round")
    
    return parser.parse_args()

def main():
    """Main function to run experiments and compare results."""
    args = parse_arguments()
    
    # Run experiments
    if args.run_all or args.run_fedavg:
        run_experiment("fedavg", args.num_clients, args.num_rounds, args.epochs)
    
    if args.run_all or args.run_fedkd:
        run_experiment("fedkd", args.num_clients, args.num_rounds, args.epochs)
    
    # Compare results
    if args.compare or args.run_all:
        compare_results()
    
    # Run comprehensive analysis
    if args.analyze:
        comprehensive_analysis()

if __name__ == "__main__":
    main() 