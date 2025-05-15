import os
import requests
from tqdm import tqdm
import psycopg2
from minio import Minio
from urllib.parse import urlparse
from dotenv import load_dotenv
from io import BytesIO

# Charger les variables d'environnement depuis .env
load_dotenv()

# Connexion PostgreSQL
conn = psycopg2.connect(
    dbname=os.getenv("POSTGRES_DB"),
    user=os.getenv("POSTGRES_USER"),
    password=os.getenv("POSTGRES_PASSWORD"),
    host=os.getenv("POSTGRES_HOST"),
    port=os.getenv("POSTGRES_PORT")
)
cursor = conn.cursor()

# Create a table if not exist
cursor.execute("""
    CREATE TABLE IF NOT EXISTS plants_data (
        id SERIAL PRIMARY KEY,
        url_source TEXT NOT NULL,
        url_s3 TEXT NOT NULL,
        label TEXT CHECK(label IN ('dandelion', 'grass')) NOT NULL
    );
""")

# Teplates of URLs
source_url_template = "https://raw.githubusercontent.com/btphan95/greenr-airflow/refs/heads/master/data/{label}/{index:08d}.jpg"
s3_url_template = "https://mlops-plants-data.s3.amazonaws.com/{label}/{index:08d}.jpg"

# Labels and index
labels = ["dandelion", "grass"]
num_images = 200  # 200 imágenes por categoría

# Insert data on PostgreSQL
for label in labels:
    for index in range(num_images):
        url_source = source_url_template.format(label=label, index=index)
        url_s3 = s3_url_template.format(label=label, index=index)
        
        # Verificar si ya existe para no duplicar
        cursor.execute(
            "SELECT 1 FROM plants_data WHERE url_s3 = %s LIMIT 1;",
            (url_s3,)
        )
        exists = cursor.fetchone()
        
        if not exists:
            cursor.execute(
                "INSERT INTO plants_data (url_source, url_s3, label) VALUES (%s, %s, %s)",
                (url_source, url_s3, label)
            )

# Valider les changements
conn.commit()

# Connexion MinIO
client = Minio(
    os.getenv("MLFLOW_S3_ENDPOINT_URL").replace("http://", ""),  # Enlève le protocole pour Minio
    access_key=os.getenv("AWS_ACCESS_KEY_ID"),
    secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    secure=False
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
    except:
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
                content_type="image/jpeg"
            )
    except Exception as e:
        print(f"Erreur : {url} => {e}")
        continue

print("✅ Toutes les images ont été envoyées dans MinIO sans stockage local.")