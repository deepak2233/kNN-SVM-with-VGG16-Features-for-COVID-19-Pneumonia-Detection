import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import seaborn as sns
from models.autoencoder import apply_autoencoder
from models.knn_svm_classifier import KNNSVMClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Streamlit app title
st.title("COVID-19 Detection from X-ray Images")

st.write("""
This app classifies X-ray images into three categories: Normal, COVID-19, or Pneumonia using a kNN-SVM model.
""")

# File uploader widget
uploaded_file = st.file_uploader("Upload an X-ray image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

def extract_features(image_array):
    model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = Model(inputs=model.input, outputs=model.layers[-1].output)
    features = model.predict(np.expand_dims(image_array, axis=0))
    return features

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying the image...")

    # Load the image and preprocess it
    image = load_img(uploaded_file, target_size=(224, 224))
    image_array = img_to_array(image)
    image_array /= 255.0  # Normalization

    # Extract features using VGG16
    features = extract_features(image_array)

    # Apply Autoencoder for dimensionality reduction
    _, reduced_features = apply_autoencoder(features, features)

    # Initialize kNN-SVM classifier and predict
    classifier = KNNSVMClassifier(k=5, kernel='linear')
    pred_class = classifier.predict(reduced_features.reshape(1, -1))  # Ensure 2D input

    # Display prediction result
    prediction_map = {0: "Normal", 1: "COVID-19", 2: "Pneumonia"}
    st.write(f"Prediction: **{prediction_map[pred_class[0]]}**")

    # Display confusion matrix (for demo purposes, here is an example)
    st.write("### Confusion Matrix")
    true_class = np.array([1])  # Example true class
    cm = confusion_matrix(true_class, pred_class)  # Replace with actual values for evaluation
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", ax=ax)
    st.pyplot(fig)

    # Classification report (for demo purposes)
    st.write("### Classification Report")
    report = classification_report(true_class, pred_class, target_names=["Normal", "COVID-19", "Pneumonia"])
    st.text(report)
