# airflow/scripts/sync_from_minio.py
import os
from minio import Minio

client = Minio(
    "minio:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
)

bucket_name = "images"
destination_dir = "/opt/airflow/data/data"
os.makedirs(destination_dir, exist_ok=True)

# Liste tous les objets du bucket
objects = client.list_objects(bucket_name, recursive=True)

for obj in objects:
    file_path = os.path.join(destination_dir, obj.object_name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    client.fget_object(bucket_name, obj.object_name, file_path)

print("✅ Images copiées depuis MinIO vers airflow/data/data")
