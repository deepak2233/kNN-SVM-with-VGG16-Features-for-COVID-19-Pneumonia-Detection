from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input
import os

def load_data(data_dir, img_size=(224, 224), batch_size=32):
    # ImageDataGenerator for augmentation on the training set
    train_datagen = ImageDataGenerator(
        rescale=1.0/255.0,  # Normalize pixel values to [0,1]
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        preprocessing_function=preprocess_input  # Preprocess using VGG16 preprocessing
    )

    # For validation/test, no augmentation but use the same preprocessing
    test_datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        preprocessing_function=preprocess_input
    )

    # Training data generator
    train_generator = train_datagen.flow_from_directory(
        directory=os.path.join(data_dir, 'train'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse'  # Class mode as sparse labels
    )

    # Validation/Test data generator
    test_generator = test_datagen.flow_from_directory(
        directory=os.path.join(data_dir, 'test'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse',
        shuffle=False  # Don't shuffle test set
    )

    return train_generator, test_generator
