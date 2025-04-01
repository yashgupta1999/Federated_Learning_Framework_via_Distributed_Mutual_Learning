import tensorflow as tf
from keras import layers
from tensorflow.keras.models import Sequential
import ast
import random
import numpy as np
from .metrics_map import map_metrics_to_tf,map_loss_to_tf,map_optimizer_to_tf

def create_model(config,seed=333,input_shape=(100,100,3)):
    """
    Creates and compiles a CNN model based on the provided configuration.
    """
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
   
    # input_shape = ast.literal_eval(CONFIG.get('image_size', '(100, 100, 3)'))
    
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
    
    model_metrics = map_metrics_to_tf(config['model_metrics'])
    model_loss = map_loss_to_tf(config['model_loss'])
    model_optimizer = map_optimizer_to_tf(config['model_optimizer'])

    model.compile(
        optimizer=model_optimizer,
        loss=model_loss,
        metrics=model_metrics
    )

    return model 