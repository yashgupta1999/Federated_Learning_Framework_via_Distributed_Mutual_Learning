import os
import numpy as np
from .image_processing import process_data_fast, save_processed_array

def load_training_data(data_path):
    """
    Loads a processed .npy data file and returns feature and label arrays.

    Args:
        fold_path (str): Path to the .npy file (e.g., 'init.npy' or 'round1.npy')

    Returns:
        tuple: (trainx, trainy)
            - trainx (np.ndarray): Feature array of shape (N, H, W, C)
            - trainy (np.ndarray): Label array of shape (N, 1)
    """
    data = np.load(data_path, allow_pickle=True)
    trainx = np.stack(data[:, 0]).astype(np.float32)
    trainy = data[:, 1].astype(int).reshape([-1, 1])
    return trainx, trainy

def process_and_save_folds(data_dir,output_dir, max_workers=8):
    """
    Processes all folds and saves them under a new directory
    automatically inferred from `data_dir`.
    """
    # Infer output_dir from parent of data_dir

    print(f"📁 Saving processed data to: {output_dir}")

    # Process init.csv
    init_path = os.path.join(data_dir, 'init.csv')
    if os.path.exists(init_path):
        processed = process_data_fast(init_path, max_workers)
        save_processed_array(processed, os.path.join(output_dir, 'init.npy'))

    # Process global folds
    global_dir = os.path.join(data_dir, 'global')
    if os.path.exists(global_dir):
        for fname in sorted(os.listdir(global_dir)):
            if fname.endswith('.csv'):
                round_idx = fname.replace('.csv', '')
                fpath = os.path.join(global_dir, fname)
                processed = process_data_fast(fpath, max_workers)
                save_processed_array(processed, os.path.join(output_dir, 'global', f"{round_idx}.npy"))

    # Process client folds
    for name in sorted(os.listdir(data_dir)):
        if name.startswith("client_"):
            client_dir = os.path.join(data_dir, name)

            for fname in sorted(os.listdir(client_dir)):
                if fname.endswith('.csv'):
                    round_idx = fname.replace('.csv', '')
                    fpath = os.path.join(client_dir, fname)
                    processed = process_data_fast(fpath, max_workers)
                    save_processed_array(processed, os.path.join(output_dir, name, f"{round_idx}.npy"))

    print(f"✅ All processed data saved under: {output_dir}") 