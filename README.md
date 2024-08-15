# COVID-19 Pneumonia Detection Using kNN-SVM and VGG16 Features

## Overview
This repository implements a novel COVID-19 Pneumonia detection system using a combination of deep feature extraction from chest X-rays via the VGG16 architecture and a kNN-SVM hybrid model for classification. This project is designed to be easily scalable for large datasets and features a flexible architecture that allows hyperparameter tuning via command-line arguments.

## Pipeline
1. **Data Preprocessing and Augmentation**: Loads and preprocesses the chest X-ray dataset, including normalization and augmentation (flipping, resizing).
2. **Exploratory Data Analysis (EDA)**: Visualizes class distributions and image samples.
3. **Feature Extraction**: Uses the pre-trained VGG16 model to extract deep features from the input images.
4. **Dimensionality Reduction**: Applies an autoencoder to reduce the dimensionality of the extracted features while preserving important patterns.
5. **Classification**: Implements a kNN-regularized SVM classifier that combines the local sensitivity of kNN with the global stability of SVMs.
6. **Evaluation**: Evaluates the model's performance using accuracy, precision, recall, and F1-score.

## Features
- **Transfer Learning**: Utilizes pre-trained VGG16 model for deep feature extraction.
- **Autoencoder**: Reduces feature dimensionality to optimize classification.
- **kNN-SVM Model**: Leverages the advantages of both k-Nearest Neighbors and Support Vector Machines for robust classification.
- **Command-line Interface**: Supports command-line arguments for batch size, number of epochs, kNN neighbors, SVM regularization parameter, and more.

## Dependencies
The required Python libraries can be installed via `requirements.txt`:

```bash
pip install -r requirements.txt
