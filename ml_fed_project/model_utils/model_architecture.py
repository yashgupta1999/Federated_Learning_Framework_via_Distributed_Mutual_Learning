import tensorflow as tf
from keras import layers
from tensorflow.keras.models import Sequential
import ast
import random
import numpy as np
from config_utils.config import CONFIG

def create_model():
    """
    Creates and compiles a CNN model based on the provided configuration.
    """
    # Set random seed for reproducibility
    seed = CONFIG.get('random_seed', 333)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
   
    input_shape = ast.literal_eval(CONFIG.get('image_size', '(100, 100, 3)'))
    
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