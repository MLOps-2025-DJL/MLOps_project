import os
from minio import Minio
from dotenv import load_dotenv

# Load environment variables (optional if already handled by Airflow)
load_dotenv()

# === Configuration ===
MODEL_FILENAME = "export.pkl"
MODEL_PATH = "/opt/airflow/scripts/saved_models/export.pkl"

MINIO_ENDPOINT = os.getenv("MLFLOW_S3_ENDPOINT_URL").replace("http://", "")
ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
BUCKET_NAME = "models"
OBJECT_NAME = MODEL_FILENAME

# === Check if model file exists ===
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Ensure it is saved during training.")

# === Connect to MinIO ===
client = Minio(
    MINIO_ENDPOINT,
    access_key=ACCESS_KEY,
    secret_key=SECRET_KEY,
    secure=False
)

# === Create bucket if it does not exist ===
if not client.bucket_exists(BUCKET_NAME):
    client.make_bucket(BUCKET_NAME)

# === Upload model to MinIO ===
client.fput_object(
    bucket_name=BUCKET_NAME,
    object_name=OBJECT_NAME,
    file_path=MODEL_PATH,
    content_type="application/octet-stream"
)

print(f"Model '{MODEL_FILENAME}' successfully uploaded to MinIO in bucket '{BUCKET_NAME}' as object '{OBJECT_NAME}'")
