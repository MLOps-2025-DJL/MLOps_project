# ✅ src/models/train.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os
import mlflow
import mlflow.tensorflow

# Tracking config
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("classification_dandelion_grass")

# Paramètres
image_size = (128, 128)
batch_size = 32
epochs = 10

# Chemin vers les données
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "", "data", "data"))
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Dossier introuvable : {data_dir}")

# Générateurs de données
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_gen = datagen.flow_from_directory(data_dir, target_size=image_size, batch_size=batch_size, class_mode="binary", subset="training")
val_gen = datagen.flow_from_directory(data_dir, target_size=image_size, batch_size=batch_size, class_mode="binary", subset="validation")

# Modèle
model = models.Sequential([
    layers.Input(shape=(*image_size, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entraînement + Tracking
with mlflow.start_run():
    history = model.fit(train_gen, validation_data=val_gen, epochs=epochs)

    # Log des paramètres
    mlflow.log_param("image_size", image_size)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("epochs", epochs)
    mlflow.log_metric("accuracy", history.history['accuracy'][-1])
    mlflow.log_metric("val_accuracy", history.history['val_accuracy'][-1])

    # Sauvegarde du modèle local et log vers MinIO (S3)
    model_path = os.path.abspath("saved_models/classifier_model")
    model.save(model_path)
    mlflow.tensorflow.log_model(tf_saved_model_dir=model_path, tf_meta_graph_tags=None, tf_signature_def_key=None, artifact_path="model")

print("\u2705 Modèle entraîné, loggé sur MLflow et sauvegardé localement.")


# ✅ src/data/download_images.py
import os
import requests
from tqdm import tqdm
import psycopg2
from minio import Minio
from urllib.parse import urlparse

# Connexion PostgreSQL
conn = psycopg2.connect(
    dbname="plants_db", user="postgres", password="postgres", host="localhost", port="5432"
)
cursor = conn.cursor()

# Connexion MinIO
client = Minio(
    "localhost:9000",
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


# ✅ airflow/dags/retrain_model_dag.py
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

with DAG(
    dag_id="retrain_model_dag",
    start_date=days_ago(1),
    schedule_interval="@daily",
    catchup=False
) as dag:

    download = BashOperator(
        task_id="download_images",
        bash_command="python /opt/airflow/scripts/download.py"
    )

    train = BashOperator(
        task_id="train_model",
        bash_command="python /opt/airflow/scripts/train.py"
    )

    download >> train
