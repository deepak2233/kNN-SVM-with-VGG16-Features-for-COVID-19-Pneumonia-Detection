import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def visualize_data_distribution(labels):
    sns.countplot(labels)
    plt.title("Distribution of Classes")
    plt.show()

def augment_image(image):
    # Example of image augmentation (horizontal flip)
    flipped_image = np.fliplr(image)
    return flipped_image

def plot_sample_images(X_train, y_train):
    plt.figure(figsize=(12, 6))
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.imshow(X_train[i])
        plt.title(y_train[i]) 
        plt.axis('off')
    plt.show()
