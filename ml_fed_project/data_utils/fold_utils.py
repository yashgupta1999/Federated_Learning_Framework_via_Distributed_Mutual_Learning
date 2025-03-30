import os
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from data_utils import create_image_dataframe

def generate_k_folds(df: pd.DataFrame, num_splits: int, random_state: int = 333):
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


def generate_stratified_k_folds(df: pd.DataFrame, num_splits: int, random_state: int = 333):
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

def save_folds(folds, num_clients, save_dir, include_global=True, stratified=False):
    """
    Saves a list of folds into a structured directory with configurable options.
    
    Args:
        folds (list[pd.DataFrame]): List of k folds.
        num_clients (int): Number of clients.
        save_dir (str): Base directory where `data/` will be created.
        include_global (bool): Whether to include a global dataset folder.
        stratified (bool): Whether the folds are stratified (for logging purposes).
    
    Returns:
        str: Path to the saved experiment data folder.
    """
    # Create main data directory
    data_dir = save_dir
    os.makedirs(data_dir, exist_ok=True)

    # Save the first fold as init.csv
    init_csv_path = os.path.join(data_dir, "init.csv")
    folds[0].to_csv(init_csv_path, index=False)
    fold_type = "Stratified" if stratified else "Regular"
    print(f"✅ Saved {fold_type} Init Fold -> {init_csv_path}")

    # Remaining folds to distribute
    remaining_folds = folds[1:]
    num_remaining_folds = len(remaining_folds)

    # Calculate partitions based on whether global dataset is included
    total_partitions = num_clients + (1 if include_global else 0)
    if total_partitions == 0:
        return data_dir
        
    folds_per_partition = num_remaining_folds // total_partitions

    current_idx = 0

    # Save Global Folds if requested
    if include_global:
        global_dir = os.path.join(data_dir, "global")
        os.makedirs(global_dir, exist_ok=True)
        for i in range(folds_per_partition):
            fold_path = os.path.join(global_dir, f"round{i+1}.csv")
            remaining_folds[current_idx + i].to_csv(fold_path, index=False)
            print(f"✅ Saved {fold_type} Global Fold {i+1} -> {fold_path}")
        current_idx += folds_per_partition

    # Save Client Folds
    for client_id in range(1, num_clients + 1):
        client_dir = os.path.join(data_dir, f"client_{client_id}")
        os.makedirs(client_dir, exist_ok=True)

        for round_num in range(folds_per_partition):
            fold_idx = current_idx + round_num
            fold_path = os.path.join(client_dir, f"round{round_num+1}.csv")
            remaining_folds[fold_idx].to_csv(fold_path, index=False)
            print(f"✅ Saved {fold_type} Client {client_id} Fold {round_num+1} -> {fold_path}")

        current_idx += folds_per_partition

    return data_dir 

def split_data(data_path, num_clients, num_rounds, save_dir, include_global=True, stratified=False):
    """
    Splits the data into folds and saves them to the specified directory.

    Args:
        data_path (str): Path to the input data CSV file.
        num_clients (int): Number of clients to split the data into.
        save_dir (str): Directory to save the folds.
        include_global (bool): Whether to include a global dataset folder.
        stratified (bool): Whether the folds are stratified (for logging purposes).

    Returns:
        str: Path to the saved experiment data folder.
    """
    # Load the data

    df = create_image_dataframe(os.path.join(data_path, 'train'), os.path.join(save_dir, 'train'), class_names=None, extensions={'.jpg', '.jpeg', '.png'})
    # Generate folds
    if include_global:
        fold_size = num_rounds * (num_clients+1) + 1
    else:
        fold_size = num_rounds * num_clients + 1

    if stratified:
        folds = generate_stratified_k_folds(df, fold_size)
    else:
        folds = generate_k_folds(df, fold_size)

    print("length of folds: ", len(folds))

    # Save the folds
    return save_folds(folds, num_clients, save_dir, include_global, stratified)
