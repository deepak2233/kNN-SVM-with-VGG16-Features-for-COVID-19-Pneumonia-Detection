import numpy as np  # Import numpy if not already done
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

def extract_vgg16_features_from_generator(generator, batch_size=32):
    # Load VGG16 model
    vgg_model = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    model = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer('block5_pool').output)

    # Calculate the steps per epoch based on the generator's size and batch size
    steps_per_epoch = int(np.ceil(generator.samples / batch_size))  # Convert to int

    # Extract features from the generator
    features = model.predict(generator, steps=steps_per_epoch, verbose=1)
    labels = generator.classes  # Extract corresponding labels

    return features, labels
