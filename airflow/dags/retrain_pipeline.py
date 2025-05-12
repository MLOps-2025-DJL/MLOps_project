from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'mlops_user',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'retrain_model_pipeline',
    default_args=default_args,
    description='Réentraîner automatiquement le modèle et le loguer dans MLflow',
    schedule_interval='@weekly',  # ou None pour manuel
    start_date=datetime(2025, 5, 10),
    catchup=False,
    tags=['mlops'],
) as dag:

    # 1. Télécharger les nouvelles images depuis PostgreSQL et les stocker dans MinIO
    download_task = BashOperator(
        task_id='download_images',
        bash_command='python3 /opt/airflow/scripts/download.py'
    )

    sync_images = BashOperator(
        task_id="sync_images_from_minio",
        bash_command="python3 /opt/airflow/scripts/sync_minio.py",
        dag=dag,
    )

    # 2. Réentraîner le modèle avec les nouvelles données
    train_task = BashOperator(
        task_id='retrain_model',
        bash_command='python3 /opt/airflow/scripts/train.py'
    )

    # (Optionnel) 3. Ajouter d’autres étapes ici comme une notification ou un déploiement

    # Définir l'ordre des tâches
    download_task >> sync_images >> train_task
