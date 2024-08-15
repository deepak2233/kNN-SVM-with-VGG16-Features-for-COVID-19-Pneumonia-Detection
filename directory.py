import os

def create_project_structure():
    # Define the directories to create
    dirs = [
        "knn_svm_covid_detection/data/train",
        "knn_svm_covid_detection/data/test",
        "knn_svm_covid_detection/models",
        "knn_svm_covid_detection/utils",
        "knn_svm_covid_detection/notebooks"
    ]
    
    # Create the directories
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)
    
    # Create the files
    files = [
        "knn_svm_covid_detection/models/vgg16_feature_extractor.py",
        "knn_svm_covid_detection/models/autoencoder.py",
        "knn_svm_covid_detection/models/knn_svm_classifier.py",
        "knn_svm_covid_detection/utils/data_loader.py",
        "knn_svm_covid_detection/utils/data_preprocessing.py",
        "knn_svm_covid_detection/notebooks/EDA.ipynb",
        "knn_svm_covid_detection/main.py",
        "knn_svm_covid_detection/README.md",
        "knn_svm_covid_detection/requirements.txt"
    ]
    
    # Create empty files
    for file in files:
        with open(file, 'w') as f:
            pass

if __name__ == "__main__":
    create_project_structure()
    print("Project structure created successfully!")

