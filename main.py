import os
import tensorflow as tf
import warnings
import argparse
import matplotlib.pyplot as plt
from utils.data_loader import load_data
from utils.data_preprocessing import visualize_data_distribution, plot_sample_images
from models.vgg16_feature_extractor import extract_vgg16_features_from_generator
from models.autoencoder import apply_autoencoder
from sklearn.metrics import classification_report, confusion_matrix
from models.knn_svm_classifier import KNNSVMClassifier  # Corrected import

# Suppress TensorFlow warnings related to GPU and minor issues
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow internal logging
warnings.filterwarnings("ignore", category=UserWarning, message=".*?CUDA.*?")  # Suppress CUDA-related warnings

# Check GPU availability
def check_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("GPU is available and will be used.")
    else:
        print("No GPU detected. Using CPU.")

# Setup MirroredStrategy for multiple GPUs
def setup_strategy():
    if len(tf.config.list_physical_devices('GPU')) > 1:
        strategy = tf.distribute.MirroredStrategy()
        print("Using MirroredStrategy for multi-GPU training.")
        return strategy
    else:
        print("Single GPU or CPU is being used.")
        return None

def main(args):
    # Check GPU availability
    check_gpu()

    # Setup strategy
    strategy = setup_strategy()

    with strategy.scope() if strategy else tf.device('/CPU:0'):
        # Load data
        print("################ Loading data using ImageDataGenerator... ########################")
        train_generator, test_generator = load_data(args.data_dir, img_size=(args.img_size, args.img_size), batch_size=args.batch_size)

        # EDA
        if args.eda:
            print("################ Performing Exploratory Data Analysis... ################")
            visualize_data_distribution(train_generator.classes)
            plot_sample_images(train_generator)

        # Feature extraction
        print("################ Extracting features using VGG16... ################")
        X_train_features, y_train = extract_vgg16_features_from_generator(train_generator, batch_size=args.batch_size)
        X_test_features, y_test = extract_vgg16_features_from_generator(test_generator, batch_size=args.batch_size)

        # Dimensionality reduction with Autoencoder
        print("################ Reducing dimensionality using Autoencoder... ################")
        X_train_reduced, X_test_reduced = apply_autoencoder(X_train_features, X_test_features, 
                                                            batch_size=args.batch_size, epochs=args.epochs)

        # Classification with kNN-SVM
        print("################ Training kNN-SVM classifier... ################")
        classifier = KNNSVMClassifier(k=args.k_neighbors, kernel='linear')
        classifier.fit(X_train_reduced, y_train)
        
        # Prediction
        print("################ Predicting on test data... ################")
        y_pred = classifier.predict(X_test_reduced)
        
        # Evaluation
        print("################ Classification Report: ################")
        print(classification_report(y_test, y_pred))
        
        print("################ Confusion Matrix: ################")
        print(confusion_matrix(y_test, y_pred))
        
        # Optionally plot loss curves
        if hasattr(classifier, 'history'):
            plt.figure()
            plt.plot(classifier.history.history['loss'], label='train_loss')
            plt.plot(classifier.history.history['val_loss'], label='val_loss')
            plt.title('Training and Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="################ kNN-SVM with VGG16 Features for COVID-19 Pneumonia Detection ################")

    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data/', help='Directory for data (with train/test folders)')
    parser.add_argument('--img_size', type=int, default=224, help='Image size for VGG16 input')

    # Model parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for feature extraction and training')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs for autoencoder training')
    parser.add_argument('--k_neighbors', type=int, default=5, help='Number of neighbors for kNN')
    parser.add_argument('--svm_c', type=float, default=1.0, help='C parameter for SVM')

    # Miscellaneous
    parser.add_argument('--eda', action='store_true', help='Perform exploratory data analysis')

    args = parser.parse_args()
    main(args)
