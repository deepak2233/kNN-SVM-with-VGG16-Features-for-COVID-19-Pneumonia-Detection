import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from models.vgg16_feature_extractor import extract_vgg16_features_from_generator
from models.autoencoder import apply_autoencoder
from models.knn_svm_classifier import KNNSVMClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit app title
st.title("COVID-19 Detection from X-ray Images")

st.write("""
This app classifies X-ray images into three categories: Normal, COVID-19, or Pneumonia using a kNN-SVM model.
""")

# File uploader widget
uploaded_file = st.file_uploader("Upload an X-ray image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying the image...")

    # Load the image and preprocess it
    image = load_img(uploaded_file, target_size=(224, 224))
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)

    # Extract features using VGG16
    features = extract_vgg16_features_from_generator(image_array)

    # Apply Autoencoder for dimensionality reduction
    _, reduced_features = apply_autoencoder(features, features)

    # Initialize kNN-SVM classifier and predict
    classifier = KNNSVMClassifier(k=5, kernel='linear')
    pred_class = classifier.predict(reduced_features)

    # Display prediction result
    if pred_class == 0:
        st.write("Prediction: **Normal**")
    elif pred_class == 1:
        st.write("Prediction: **COVID-19**")
    else:
        st.write("Prediction: **Pneumonia**")

    # Display confusion matrix (for demo purposes, here is an example)
    st.write("### Confusion Matrix")
    true_class = [1]  # Example true class
    cm = confusion_matrix(true_class, pred_class)  # Replace with actual values for evaluation
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", ax=ax)
    st.pyplot(fig)

    # Classification report (for demo purposes)
    st.write("### Classification Report")
    report = classification_report(true_class, pred_class, target_names=["Normal", "COVID-19", "Pneumonia"])
    st.text(report)
