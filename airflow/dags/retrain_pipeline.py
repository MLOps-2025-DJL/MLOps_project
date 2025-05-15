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
    description='Pipeline de réentraînement avec MLflow + MinIO',
    schedule_interval='@weekly',
    start_date=datetime(2025, 5, 10),
    catchup=False,
    tags=['mlops'],
) as dag:

    insert_metadata = BashOperator(
        task_id='insert_metadata_to_postgres',
        bash_command='python3 /opt/airflow/scripts/data/insert_metadata_to_postgres.py'
    )

    download_images = BashOperator(
        task_id='download_images',
        bash_command='python3 /opt/airflow/scripts/download.py'
    )

    retrain_model = BashOperator(
        task_id='retrain_model',
        bash_command='python3 /opt/airflow/scripts/train.py'
    )

    save_model = BashOperator(
        task_id='save_model_to_minio',
        bash_command='python3 /opt/airflow/scripts/save_model.py'
    )

    redeploy_model = BashOperator(
        task_id='redeploy_model',
        bash_command='curl -X POST http://api:5000/reload || echo "No redeploy endpoint available."'
    )

    insert_metadata >> download_images >> retrain_model >> save_model >> redeploy_model
