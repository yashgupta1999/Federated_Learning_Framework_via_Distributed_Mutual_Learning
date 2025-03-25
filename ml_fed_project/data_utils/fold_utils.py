import os
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from config_utils.config import CONFIG

def generate_k_folds(df: pd.DataFrame, num_splits: int, random_state: int = CONFIG['random_seed']):
    """
    Splits a DataFrame into k stratified (or regular) folds.

    Args:
        df (pd.DataFrame): The input dataframe to split.
        num_splits (int): Number of folds.
        random_state (int): Seed for reproducible shuffling.

    Returns:
        List[pd.DataFrame]: A list of DataFrames, each representing one fold.
    """
    if num_splits < 2:
        raise ValueError("num_splits must be at least 2")

    shuffled_df = shuffle(df, random_state=random_state)
    indices = np.array_split(shuffled_df.index, num_splits)
    return [shuffled_df.loc[idx].reset_index(drop=True) for idx in indices]

def generate_stratified_k_folds(df: pd.DataFrame, num_splits: int, random_state: int = CONFIG['random_seed']):
    """
    Splits a DataFrame into stratified k folds based on the label column.

    Args:
        df (pd.DataFrame): The input dataframe to split (must have a 'label' column).
        num_splits (int): Number of folds.
        random_state (int): Seed for reproducible splits.

    Returns:
        List[pd.DataFrame]: A list of DataFrames, each representing one stratified fold.
    """
    if 'label' not in df.columns:
        raise ValueError("DataFrame must contain a 'label' column for stratified splitting.")

    if num_splits < 2:
        raise ValueError("num_splits must be at least 2")

    skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=random_state)
    folds = []

    for _, test_idx in skf.split(df, df['label']):
        fold_df = df.iloc[test_idx].reset_index(drop=True)
        folds.append(fold_df)

    return folds

def save_folds_with_global(folds, num_clients, save_dir):
    """
    Saves a list of folds into a structured directory with:
    - `data/init.csv` for the first odd fold.
    - `data/global/` containing an equal number of folds as each client.
    - `data/client_X/` folders each containing an equal number of folds.

    Args:
        folds (list[pd.DataFrame]): List of k folds.
        num_clients (int): Number of clients.
        save_dir (str): Base directory where `data/` will be created.

    Returns:
        str: Path to the saved experiment data folder.
    """
    # Create main data directory
    data_dir = os.path.join(save_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # Save the first fold as init.csv
    init_csv_path = os.path.join(data_dir, "init.csv")
    folds[0].to_csv(init_csv_path, index=False)
    print(f"✅ Saved Init Fold -> {init_csv_path}")

    # Remaining folds to distribute
    remaining_folds = folds[1:]
    num_remaining_folds = len(remaining_folds)

    # Total partitions (clients + global)
    total_partitions = num_clients + 1  # Clients + Global
    folds_per_partition = num_remaining_folds // total_partitions  # Equal distribution

    # Save Global Folds
    global_dir = os.path.join(data_dir, "global")
    os.makedirs(global_dir, exist_ok=True)
    for i in range(folds_per_partition):
        fold_path = os.path.join(global_dir, f"round{i+1}.csv")
        remaining_folds[i].to_csv(fold_path, index=False)
        print(f"✅ Saved Global Fold {i+1} -> {fold_path}")

    # Save Client Folds
    current_idx = folds_per_partition
    for client_id in range(1, num_clients + 1):
        client_dir = os.path.join(data_dir, f"client_{client_id}")
        os.makedirs(client_dir, exist_ok=True)

        for round_num in range(folds_per_partition):
            fold_idx = current_idx + round_num
            fold_path = os.path.join(client_dir, f"round{round_num+1}.csv")
            remaining_folds[fold_idx].to_csv(fold_path, index=False)
            print(f"✅ Saved Client {client_id} Fold {round_num+1} -> {fold_path}")

        current_idx += folds_per_partition

    return data_dir  # Return the path for reference 