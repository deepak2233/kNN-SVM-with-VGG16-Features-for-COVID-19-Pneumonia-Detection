from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

def extract_vgg16_features(X_train, X_test, batch_size=32):
    # Load the VGG16 model with pre-trained ImageNet weights
    vgg_model = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    model = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer('block5_pool').output)
    
    # Extract features with the specified batch size
    X_train_features = model.predict(X_train, batch_size=batch_size)
    X_test_features = model.predict(X_test, batch_size=batch_size)
    
    return X_train_features, X_test_features
