from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import joblib
import io
from PIL import Image  # Ensure this is imported
from tensorflow.keras.losses import MeanSquaredError

# Load the models with custom objects
autoencoder = load_model('saved_model/autoencoder.h5', custom_objects={'mse': MeanSquaredError()})
classifier = joblib.load('saved_model/knn_svm_classifier.joblib')

app = FastAPI()

def preprocess_image(image_data):
    try:
        # Convert the uploaded file to a PIL Image
        image = Image.open(image_data)

        # Resize and preprocess the image as required by VGG16
        image = image.resize((224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = image / 255.0  # Normalize
        return image
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded file as bytes
        image_data = io.BytesIO(await file.read())
        
        # Preprocess the image
        image = preprocess_image(image_data)
        
        # Extract features using VGG16 part of the autoencoder
        features = autoencoder.layers[0](image)
        
        # Use the encoder part of the autoencoder to reduce dimensionality
        features_reduced = autoencoder.predict(features)
        
        # Predict using the kNN-SVM classifier
        prediction = classifier.predict(features_reduced)
        
        # Convert prediction to class label
        labels = ["Normal", "COVID-19", "Pneumonia"]
        predicted_class = labels[prediction[0]]
        
        return JSONResponse(content={"prediction": predicted_class})
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
