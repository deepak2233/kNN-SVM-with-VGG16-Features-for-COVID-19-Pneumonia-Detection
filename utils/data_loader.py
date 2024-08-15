import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split

def load_data(data_dir, img_size=(224, 224)):
    images = []
    labels = []
    
    # Assuming the dataset is in 'Normal', 'COVID', 'Pneumonia' subdirectories
    for label in ['Normal', 'COVID', 'Pneumonia']:
        path = os.path.join(data_dir, label)
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, img_size)
            img = img_to_array(img)
            images.append(img)
            labels.append(label)
    
    images = np.array(images) / 255.0  # Normalize the images
    labels = np.array(labels)
    
    return train_test_split(images, labels, test_size=0.2, random_state=42)
