from fastapi import FastAPI, UploadFile, File, HTTPException
from fastai.vision.all import load_learner
from PIL import Image
import numpy as np
import io
import os
import logging
from dotenv import load_dotenv
import logging
import torchvision.transforms as transforms
import torch
import boto3
from io import BytesIO
import pickle
from datetime import datetime

logging.basicConfig(level=logging.INFO)

# Chargement des variables d'environnement
load_dotenv()

app = FastAPI()

model = None

def find_latest_model():
    # Connexion à MinIO
    s3 = boto3.client('s3',
                      endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL"), 
                      aws_access_key_id=os.getenv("MINIO_ROOT_USER"),
                      aws_secret_access_key=os.getenv("MINIO_ROOT_PASSWORD"))
    
    latest_model_info = None

    # Lister tous les buckets
    buckets = s3.list_buckets()['Buckets']
    for bucket in buckets:
        bucket_name = bucket['Name']
        logging.info(f"Exploring bucket: {bucket_name}")
        
        # Lister les objets dans le bucket
        objects = s3.list_objects_v2(Bucket=bucket_name)
        if 'Contents' in objects:
            for obj in objects['Contents']:
                key = obj['Key']
                if key.endswith('.pkl'):  # Vérifier si c'est un fichier .pkl
                    last_modified = obj['LastModified']
                    if latest_model_info is None or last_modified > latest_model_info['LastModified']:
                        latest_model_info = {
                            'Bucket': bucket_name,
                            'Key': key,
                            'LastModified': last_modified
                        }
    
    if latest_model_info is None:
        raise FileNotFoundError("No .pkl model files found in any bucket.")
    
    logging.info(f"Latest model found: {latest_model_info}")
    return latest_model_info

def load_model():
    # Trouver le modèle le plus récent
    latest_model_info = find_latest_model()
    bucket_name = latest_model_info['Bucket']
    key = latest_model_info['Key']

    # Télécharger le fichier .pkl
    s3 = boto3.client('s3',
                      endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL"), 
                      aws_access_key_id=os.getenv("MINIO_ROOT_USER"),
                      aws_secret_access_key=os.getenv("MINIO_ROOT_PASSWORD"))
    obj = s3.get_object(Bucket=bucket_name, Key=key)
    model_bytes = BytesIO(obj['Body'].read())

    return load_learner(model_bytes)

def predict_image(image, model):
    class_name, _, probs = model.predict(image)
    confidence = probs.max().item()
    return class_name, confidence

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global model

    if model is None:
        try:
            model = load_model()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not load model: {str(e)}")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    class_name, confidence = predict_image(image, model)
    return {"prediction": class_name, "probability": confidence}