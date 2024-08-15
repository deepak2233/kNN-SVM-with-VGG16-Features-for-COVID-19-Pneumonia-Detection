import argparse
from utils.data_loader import load_data
from utils.data_preprocessing import visualize_data_distribution, plot_sample_images
from models.vgg16_feature_extractor import extract_vgg16_features
from models.autoencoder import apply_autoencoder
from models.knn_svm_classifier import knn_svm_classifier

def main(args):
    # Load data
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data(args.data_dir, img_size=(args.img_size, args.img_size))

    # EDA
    if args.eda:
        print("Performing Exploratory Data Analysis...")
        visualize_data_distribution(y_train)
        plot_sample_images(X_train, y_train)

    # Feature extraction
    print("Extracting features using VGG16...")
    X_train_features, X_test_features = extract_vgg16_features(X_train, X_test, batch_size=args.batch_size)

    # Dimensionality reduction with Autoencoder
    print("Reducing dimensionality using Autoencoder...")
    X_train_reduced, X_test_reduced = apply_autoencoder(X_train_features, X_test_features, 
                                                        batch_size=args.batch_size, epochs=args.epochs)

    # Classification with kNN-SVM
    print("Training kNN-SVM classifier...")
    knn_svm_classifier(X_train_reduced, X_test_reduced, y_train, y_test, k=args.k_neighbors, c=args.svm_c)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="kNN-SVM with VGG16 Features for COVID-19 Pneumonia Detection")

    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data/train/', help='Directory for training data')
    parser.add_argument('--img_size', type=int, default=224, help='Image size for VGG16 input')

    # Model parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for feature extraction and training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for autoencoder training')
    parser.add_argument('--k_neighbors', type=int, default=5, help='Number of neighbors for kNN')
    parser.add_argument('--svm_c', type=float, default=1.0, help='C parameter for SVM')

    # Miscellaneous
    parser.add_argument('--eda', action='store_true', help='Perform exploratory data analysis')

    args = parser.parse_args() 
    main(args)
