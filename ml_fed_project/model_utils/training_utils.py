def train_model(model, trainx, trainy, epochs=10, 
                batch_size=16, verbose=1, validation_data=None):
    """
    Trains a compiled Keras model.
    """
    history = model.fit(
        trainx,
        trainy,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        validation_data=validation_data
    )
    return history 