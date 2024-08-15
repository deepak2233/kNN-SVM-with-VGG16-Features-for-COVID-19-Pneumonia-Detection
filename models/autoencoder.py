from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.keras.models import Sequential

def autoencoder_model(input_shape):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Reshape(input_shape))
    
    return model

def apply_autoencoder(X_train, X_test, batch_size=32, epochs=10):
    autoencoder = autoencoder_model(X_train.shape[1:])
    autoencoder.compile(optimizer='adam', loss='mse')
    
    # Train autoencoder
    autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, X_test))
    
    # Use the encoder part of the autoencoder to reduce dimensionality
    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('dense_2').output)
    X_train_reduced = encoder.predict(X_train, batch_size=batch_size)
    X_test_reduced = encoder.predict(X_test, batch_size=batch_size)
    
    return X_train_reduced, X_test_reduced
