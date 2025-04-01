import tensorflow as tf
from model_utils.metrics_map import map_metrics_to_tf

def train_model(config, model, trainx, trainy, epochs=10, 
                batch_size=16, verbose=1, validation_data=None):
    """
    Trains a compiled Keras model.
    """
    return model.fit(
        trainx, trainy,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        validation_data=validation_data
    )

def train_fedprox(config, model, trainx, trainy, global_weights, loss_fn, optimizer, 
                   mu=0.1, epochs=5, batch_size=16):
    """
    Trains the model using FedProx custom training loop.
    """
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((trainx, trainy))\
        .shuffle(buffer_size=1024)\
        .batch(batch_size)
    
    # Initialize history with metrics
    history = {metric: [] for metric in config['model_metrics']}
    history['loss'] = []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        epoch_loss = tf.keras.metrics.Mean()

        for step, (x_batch, y_batch) in enumerate(dataset):
            with tf.GradientTape() as tape:
                y_pred = model(x_batch, training=True)
                base_loss = loss_fn(y_batch, y_pred)
                
                # Compute proximal term more efficiently
                prox_loss = tf.add_n([
                    tf.reduce_sum(tf.square(v - gw))
                    for v, gw in zip(model.trainable_variables, global_weights)
                ])
                
                total_loss = base_loss + 0.5 * mu * prox_loss

            # Update weights
            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss.update_state(total_loss)

            if step % 100 == 0:
                print(f"  Step {step}: total loss = {total_loss:.4f}")

        avg_loss = epoch_loss.result()
        print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")
        history["loss"].append(float(avg_loss))
        for metric in map_metrics_to_tf(config['model_metrics']):
            metric.update_state(trainy, y_pred)
            history[metric.name].append(float(metric.result()))
        epoch_loss.reset_states()

    return model, history