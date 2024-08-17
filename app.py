import streamlit as st
import requests
from PIL import Image
import io

# Streamlit app title
st.title("COVID-19 Detection from X-ray Images")

st.write("""
This app allows you to upload an X-ray image and get predictions on whether it's Normal, COVID-19, or Pneumonia.
""")

# File uploader widget
uploaded_file = st.file_uploader("Upload an X-ray image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying the image...")

    # Convert the uploaded file to an io.BytesIO object
    img_bytes = io.BytesIO(uploaded_file.read())

    # Send the image to FastAPI for prediction
    files = {"file": img_bytes}
    response = requests.post("http://localhost:8000/predict", files=files)
    
    if response.status_code == 200:
        result = response.json()
        st.write(f"Prediction: **{result['prediction']}**")
    else:
        st.write(f"Error: {response.content.decode()}")
