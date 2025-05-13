from fastapi import FastAPI, UploadFile, File
from PIL import Image
import numpy as np
import io
import logging
import torchvision.transforms as transforms
import torch
import boto3
from io import BytesIO
from tensorflow.keras.models import load_model as keras_load_model

logging.basicConfig(level=logging.INFO)

def load_model():
    # Connexion à MinIO
    s3 = boto3.client('s3',
                      endpoint_url='http://minio:9000', 
                      aws_access_key_id='minioadmin',
                      aws_secret_access_key='minioadmin')
    
    # Télécharger le fichier .h5
    obj = s3.get_object(Bucket='models', Key='saved_models/classifier_model.h5')
    model_bytes = BytesIO(obj['Body'].read())

    # Sauvegarder temporairement (nécessaire pour load_model)
    temp_path = '/tmp/model.h5'
    with open(temp_path, 'wb') as f:
        f.write(model_bytes.getbuffer())

    # Charger avec TensorFlow
    model = keras_load_model(temp_path)
    return model

def predict_image(image, model):
    # Redimensionner l'image à 128x128 (taille attendue par le modèle)
    image = image.resize((128, 128))
    img_array = np.array(image) / 255.0  # Normalisation entre 0 et 1
    img_array = np.expand_dims(img_array, axis=0)  # Ajouter une dimension batch

    # Effectuer la prédiction avec le modèle Keras
    output = model.predict(img_array)  # Probabilité pour que ce soit de l'herbe
    predicted_probabilities = output[0] 

    # Retourner la classe prédite et la probabilité
    if predicted_probabilities >= 0.5:
        class_name = "grass"
    else:
        class_name = "dandelion"
        predicted_probabilities = 1 - predicted_probabilities

    return class_name, predicted_probabilities
app = FastAPI()

model = load_model()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    class_name, probabilities = predict_image(image, model)
    return {
        "prediction": class_name,
        "probabilities": probabilities.tolist()  # Convertir en liste pour JSON
    }

