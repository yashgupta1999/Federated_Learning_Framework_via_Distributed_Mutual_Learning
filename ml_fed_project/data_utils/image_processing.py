import os
import numpy as np
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from tensorflow.keras.utils import load_img
from tensorflow.keras.preprocessing.image import img_to_array

def process_image(image_path: str, label: str, input_shape: tuple = (100, 100, 3)) -> tuple:
    """
    Process a single image for model training/inference.
    
    Args:
        image_path (str): Path to the image file
        label (str): Image label ('mask' or 'no_mask')
    
    Returns:
        tuple: Processed image array and class label (1 for mask, 0 for no_mask)
        None: If processing fails
    """
    try:
        image = load_image(image_path, target_size=(input_shape[0], input_shape[0]))
        image_class = 1 if label == 'mask' else 0
        return (image, image_class)
    except Exception as e:
        print(f"Error processing image at {image_path}: {e}")
        return None

def process_image_safe(image_path: str, label: str) -> tuple:
    """
    Wrapper function for process_image with additional error handling.
    
    Args:
        image_path (str): Path to the image file
        label (str): Image label ('mask' or 'no_mask')
    
    Returns:
        tuple: Processed image array and class label
        None: If processing fails
    """
    try:
        return process_image(image_path, label)
    except Exception as e:
        print(f"❌ Error processing {image_path}: {e}")
        return None

def process_data_fast(data_path: str, max_workers: int = 8) -> np.ndarray:
    """
    Parallel processing of multiple images using ThreadPoolExecutor.
    
    Args:
        data_path (str): Path to CSV file containing image filepaths and labels
        max_workers (int): Maximum number of concurrent threads for processing
    
    Returns:
        np.ndarray: Array of processed images and their labels
    """
    import pandas as pd
    df = pd.read_csv(data_path)
    filepaths = df['filepath'].tolist()
    labels = df['label'].tolist()
    processed = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_image_safe, path, label): path 
                  for path, label in zip(filepaths, labels)}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                processed.append(result)
    return np.array(processed, dtype=object)

def save_processed_array(array: np.ndarray, output_path: str) -> None:
    """
    Save processed image array to disk.
    
    Args:
        array (np.ndarray): Array of processed images and labels
        output_path (str): Path where the array will be saved
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, array)
    print(f"💾 Saved processed data to {output_path}")

def load_image(image_path: str, target_size: tuple = (100, 100)) -> np.ndarray:
    """
    Load and preprocess an image from the given path.
    
    Args:
        image_path (str): Path to the image file
        target_size (tuple): Desired dimensions for the output image (height, width)
    
    Returns:
        np.ndarray: Preprocessed image array normalized to [0, 1]
        
    Raises:
        Exception: If image loading or processing fails
    """
    try:
        # Load the image and resize to target dimensions
        image = load_img(image_path, target_size=target_size)
        # Convert to numpy array
        image_array = img_to_array(image)
        # Normalize pixel values to [0, 1]
        normalized_image = image_array / 255.0
        
        return normalized_image
    except Exception as e:
        raise Exception(f"Failed to load image from {image_path}: {str(e)}")