import os
import pandas as pd
from pathlib import Path
from sklearn.utils import shuffle

def create_image_dataframe(data_dir, output_dir, class_names=None, extensions={'.jpg', '.jpeg', '.png'}):
    """
    Scans a directory with subfolders for each class and returns a DataFrame
    with image file paths and corresponding class labels.

    Args:
        data_dir (str or Path): Root directory containing class subfolders.
        class_names (list[str], optional): If provided, filters only these subfolders.
        extensions (set): Allowed image file extensions.

    Returns:
        pd.DataFrame: DataFrame with columns ['filepath', 'label']
    """
    data = []
    data_dir = Path(data_dir).resolve()  # Ensures it's absolute
    print(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {data_dir}")

    for class_folder in data_dir.iterdir():
        if class_folder.is_dir():
            label = class_folder.name
            if class_names and label not in class_names:
                continue

            for file in class_folder.glob("*"):
                if file.suffix.lower() in extensions:
                    data.append((str(file), label))

    df = pd.DataFrame(data, columns=["filepath", "label"])
    df.to_csv(output_dir, index=False)
    return shuffle(df).reset_index(drop=True) 