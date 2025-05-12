import os
import requests
from tqdm import tqdm
import psycopg2
from minio import Minio
from urllib.parse import urlparse

# Connexion PostgreSQL
conn = psycopg2.connect(
    dbname="plants_db", user="postgres", password="postgres", host="postgres", port="5432"
)
cursor = conn.cursor()

# Connexion MinIO
client = Minio(
    "minio:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
)

# Bucket d’images
bucket_name = "images"
if not client.bucket_exists(bucket_name):
    client.make_bucket(bucket_name)

# Répertoire temporaire de téléchargement
download_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "", "data", "downloaded"))
os.makedirs(download_dir, exist_ok=True)

cursor.execute("SELECT id, url_source, label FROM plants_data;")
data = cursor.fetchall()

for row in tqdm(data):
    img_id, url, label = row
    filename = f"{label}/{img_id:08d}.jpg"
    local_path = os.path.join(download_dir, filename)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # Téléchargement image
    if not os.path.exists(local_path):
        try:
            r = requests.get(url, timeout=5)
            with open(local_path, 'wb') as f:
                f.write(r.content)
        except Exception as e:
            print(f"Erreur de téléchargement : {url} => {e}")
            continue

    # Upload MinIO
    client.fput_object(bucket_name, filename, local_path)

print("\u2705 Toutes les images ont été téléchargées et uploadées sur MinIO")
