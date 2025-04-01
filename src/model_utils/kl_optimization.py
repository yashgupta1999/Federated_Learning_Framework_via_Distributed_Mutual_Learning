import tensorflow as tf
import numpy as np

class KLLoss(tf.keras.losses.Loss):
    """Custom KL divergence loss class for ensemble distillation."""
    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        KL = []
        # Using tf.keras.losses.KLDivergence to compute KL divergence for each teacher prediction
        kl = tf.keras.losses.KLDivergence()
        for i in y_pred:
            KL.append(kl(y_true, i))
        KL = tf.convert_to_tensor(KL, dtype=tf.float32)
        return tf.reduce_mean(KL)

def loss(model, x, y, training, preds):
    """Combined loss function with Binary Crossentropy and KL divergence."""
    # Forward pass through the model
    y_ = model(x, training=training)
    ceLoss = tf.keras.losses.BinaryCrossentropy()(y, y_)
    
    # Compute the KL loss between the model output and the teacher predictions
    loss_object = KLLoss()
    return ceLoss + loss_object(y_true=y_, y_pred=preds)

def grad(model, inputs, targets, preds):
    """Compute gradients using GradientTape."""
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, True, preds)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

def ceil(a, b):
    """Helper function for ceiling division to compute number of batches."""
    return -(-a // b)

def optimize_weights(model, X, Y, preds, batch_size=32, num_epochs=10):
    """
    Optimize model weights using a custom training loop with KL divergence loss.
    
    Args:
        model: TensorFlow model to optimize
        X: Input features
        Y: Target labels
        preds: Teacher predictions (3D array for multiple teacher predictions)
        batch_size: Size of training batches (default: 32)
        num_epochs: Number of training epochs (default: 10)
    
    Returns:
        tuple: (optimized model, accuracy history, loss history)
    """
    optimizer = tf.keras.optimizers.Adam()
    n_samples = len(X)
    batches = ceil(n_samples, batch_size)
    
    # Split the data into batches
    batchx = np.array_split(X, batches)
    batchy = np.array_split(Y, batches)
    batchpreds = np.array_split(np.array(preds), batches, axis=1)
    
    train_loss_results = []
    train_accuracy_results = []
    
    for epoch in range(num_epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.BinaryAccuracy()
        
        # Iterate over each batch
        for x, y, z in zip(batchx, batchy, batchpreds):
            loss_value, grads = grad(model, x, y, z)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss_avg.update_state(loss_value)
            epoch_accuracy.update_state(y, model(x, training=True))
        
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(
            epoch, epoch_loss_avg.result(), epoch_accuracy.result()))
    
    accHistory = [i.numpy() for i in train_accuracy_results]
    lossHistory = [i.numpy() for i in train_loss_results]
    
    return model, accHistory, lossHistory 