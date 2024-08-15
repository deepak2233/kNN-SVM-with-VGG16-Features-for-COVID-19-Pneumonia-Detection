import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

def load_data(data_dir, img_size=(224, 224)):
    images = []
    labels = []
    class_labels = ['Normal', 'COVID', 'Pneumonia']  # Define your class labels
    
    # Check if the data directory exists
    if not os.path.exists(data_dir):
        raise ValueError(f"The directory {data_dir} does not exist.")
    
    # Loop through the subdirectories
    for label in class_labels:
        label_path = os.path.join(data_dir, label)
        
        # Check if the class directory exists
        if not os.path.exists(label_path):
            print(f"Warning: The directory for {label} does not exist. Skipping.")
            continue
        
        # Loop through the images in the class directory
        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)
            try:
                img = cv2.imread(img_path)
                
                # Check if image was loaded correctly
                if img is None:
                    print(f"Warning: Unable to load image {img_path}. Skipping.")
                    continue
                
                img = cv2.resize(img, img_size)
                img = img_to_array(img)
                images.append(img)
                labels.append(label)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}. Skipping.")
    
    # Normalize images
    images = np.array(images, dtype="float32") / 255.0
    
    # Encode labels
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    labels = to_categorical(labels)
    
    # Split the data
    return train_test_split(images, labels, test_size=0.2, random_state=42)

# Example usage:
# X_train, X_test, y_train, y_test = load_data('/path/to/dataset')
