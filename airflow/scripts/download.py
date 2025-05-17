import os
import requests
from tqdm import tqdm
import psycopg2
from minio import Minio
from io import BytesIO

# Connexion PostgreSQL
conn = psycopg2.connect(
    dbname=os.getenv("POSTGRES_DB"),
    user=os.getenv("POSTGRES_USER"),
    password=os.getenv("POSTGRES_PASSWORD"),
    host=os.getenv("POSTGRES_HOST"),
    port=os.getenv("POSTGRES_PORT"),
)
cursor = conn.cursor()

# Connexion MinIO
client = Minio(
    os.getenv("MLFLOW_S3_ENDPOINT_URL").replace(
        "http://", ""
    ),  # Enlève le protocole pour Minio
    access_key=os.getenv("AWS_ACCESS_KEY_ID"),
    secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    secure=False,
)

# Bucket d’images
bucket_name = "images"
if not client.bucket_exists(bucket_name):
    client.make_bucket(bucket_name)

# Récupération des données
cursor.execute("SELECT id, url_source, label FROM plants_data;")
data = cursor.fetchall()

# Upload direct dans MinIO sans stockage local
for row in tqdm(data):
    img_id, url, label = row
    object_name = f"{label}/{img_id:08d}.jpg"

    # Vérifie si l'objet existe déjà
    found = False
    try:
        client.stat_object(bucket_name, object_name)
        found = True
    except Exception:
        pass

    if found:
        continue

    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            image_stream = BytesIO(response.content)
            client.put_object(
                bucket_name,
                object_name,
                image_stream,
                length=len(response.content),
                content_type="image/jpeg",
            )
    except Exception as e:
        print(f"Erreur : {url} => {e}")
        continue

print("✅ Toutes les images ont été envoyées dans MinIO sans stockage local.")
