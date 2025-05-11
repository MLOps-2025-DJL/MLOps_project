""from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import os
import zipfile
import boto3
import psycopg2
from datetime import timedelta
from PIL import Image
from io import BytesIO
import requests

# DAG Configuration
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# AWS S3 Configuration
BUCKET_NAME = 'mlops-plants-data'
PREPROCESS_PATH = '/opt/airflow/data/preprocess'
ZIP_FILE = '/opt/airflow/data/preprocess.zip'

# PostgreSQL Configuration
POSTGRES_CONFIG = {
    'dbname': 'plants_db',
    'user': 'postgres',
    'password': 'password',
    'host': 'localhost',
    'port': '5432'
}

# Initialize S3 client
s3 = boto3.client('s3', aws_access_key_id='minioadmin', aws_secret_access_key='minioadmin', endpoint_url='http://minio:9000')

def download_images():
    os.makedirs(PREPROCESS_PATH, exist_ok=True)
    
    conn = psycopg2.connect(**POSTGRES_CONFIG)
    cursor = conn.cursor()
    cursor.execute("SELECT url_s3, label FROM plants_data")
    images = cursor.fetchall()

    for url, label in images:
        response = requests.get(url)
        if response.status_code == 200:
            image_name = os.path.join(PREPROCESS_PATH, f"{label}_{url.split('/')[-1]}")
            with open(image_name, 'wb') as f:
                f.write(response.content)
    cursor.close()
    conn.close()

def zip_images():
    with zipfile.ZipFile(ZIP_FILE, 'w') as zipf:
        for root, _, files in os.walk(PREPROCESS_PATH):
            for file in files:
                zipf.write(os.path.join(root, file), file)

def upload_to_s3():
    s3.upload_file(ZIP_FILE, BUCKET_NAME, 'preprocessed/preprocess.zip')

with DAG(
    'image_preprocessing_pipeline',
    default_args=default_args,
    description='A DAG to preprocess images and upload to S3',
    schedule_interval=timedelta(days=1),
    start_date=days_ago(1),
    catchup=False,
) as dag:

    download_task = PythonOperator(
        task_id='download_images',
        python_callable=download_images
    )

    zip_task = PythonOperator(
        task_id='zip_images',
        python_callable=zip_images
    )

    upload_task = PythonOperator(
        task_id='upload_to_s3',
        python_callable=upload_to_s3
    )

    download_task >> zip_task >> upload_task
    ""
