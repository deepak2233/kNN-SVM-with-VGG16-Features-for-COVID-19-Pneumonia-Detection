import numpy as np
from tensorflow.keras.layers import Dense, Flatten, Reshape, InputLayer
from tensorflow.keras.models import Sequential, Model  
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten

def autoencoder_model(input_shape):
    # Calculate the number of units to match the target shape
    flattened_dim = np.prod(input_shape)  

    # Autoencoder architecture
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    model.add(Flatten())  # Flatten the input images

    # Encoder part
    model.add(Dense(1024, activation='relu', name='dense_1'))  # Intermediate dense layer
    model.add(Dense(512, activation='relu', name='dense_2'))
    model.add(Dense(256, activation='relu', name='dense_3'))  # Reduced dimensionality
    model.add(Dense(128, activation='relu', name='dense_4'))  # Bottleneck layer

    # Decoder part (expand back to the original flattened shape)
    model.add(Dense(256, activation='relu', name='dense_5'))  # Start expanding
    model.add(Dense(512, activation='relu', name='dense_6'))
    model.add(Dense(1024, activation='relu', name='dense_7'))
    model.add(Dense(flattened_dim, activation='sigmoid', name='dense_8'))  # Final dense layer before reshaping

    # Reshape back to the original image structure
    model.add(Reshape(input_shape))  
    model.summary()
    return model



def apply_autoencoder(X_train, X_test, batch_size=32, epochs=10):
    # Build the autoencoder model
    autoencoder = autoencoder_model(X_train.shape[1:])
    autoencoder.compile(optimizer='adam', loss='mse')  # Mean Squared Error (MSE) for reconstruction

    print(f"Shape of X_train before autoencoder: {X_train.shape}")
    print(f"Shape of X_test before autoencoder: {X_test.shape}")

    # Train the autoencoder (unsupervised learning)
    history = autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, X_test))

    # Check if the input size is consistent after training
    print(f"Shape of autoencoder input after training: {X_train.shape}")

    # Define the encoder model with explicit input shape
    input_tensor = Input(shape=X_train.shape[1:])
    x = Flatten()(input_tensor)  # Flatten the input
    for i in range(4):  # Apply the first 4 dense layers from the autoencoder
        x = autoencoder.layers[i+1](x)
    encoder = Model(inputs=input_tensor, outputs=x)

    # Extract compressed features
    X_train_reduced = encoder.predict(X_train, batch_size=batch_size)
    X_test_reduced = encoder.predict(X_test, batch_size=batch_size)

    print(f"Shape of X_train_reduced after autoencoder: {X_train_reduced.shape}")
    print(f"Shape of X_test_reduced after autoencoder: {X_test_reduced.shape}")

    return X_train_reduced, X_test_reduced, autoencoder


