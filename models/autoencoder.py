from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.keras.models import Sequential, Model

def autoencoder_model(input_shape):
    # Autoencoder architecture
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))  # Flatten the input images
    model.add(Dense(256, activation='relu', name='dense_1'))  # Encoder part
    model.add(Dense(128, activation='relu', name='dense_2'))
    model.add(Dense(64, activation='relu', name='dense_3'))  # Bottleneck layer (compression)
    model.add(Dense(128, activation='relu', name='dense_4'))  # Decoder part
    model.add(Dense(256, activation='relu', name='dense_5'))
    model.add(Reshape(input_shape))  # Reshape back to original input shape
    
    return model

def apply_autoencoder(X_train, X_test, batch_size=32, epochs=10):
    # Build the autoencoder model
    autoencoder = autoencoder_model(X_train.shape[1:])
    autoencoder.compile(optimizer='adam', loss='mse')  # Mean Squared Error (MSE) for reconstruction

    # Train the autoencoder (unsupervised learning)
    autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, X_test))

    # Create encoder model (up to the bottleneck layer for dimensionality reduction)
    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('dense_3').output)

    # Extract compressed features
    X_train_reduced = encoder.predict(X_train, batch_size=batch_size)
    X_test_reduced = encoder.predict(X_test, batch_size=batch_size)
    
    return X_train_reduced, X_test_reduced
