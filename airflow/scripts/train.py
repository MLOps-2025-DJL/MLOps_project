from fastai.vision.all import (
    ImageDataLoaders,
    Resize,
    cnn_learner,
    resnet34,
    accuracy,
)
from minio import Minio
import os
import shutil
import mlflow
from sklearn.metrics import f1_score, precision_score, recall_score
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

# Chargement des variables d'environnement
load_dotenv()


def download_minio_dataset(bucket_name="images", local_dir="/tmp/images"):
    print("⬇️ Téléchargement des données depuis MinIO...")

    endpoint = os.getenv("MLFLOW_S3_ENDPOINT_URL").replace("http://", "")
    minio_client = Minio(
        endpoint,
        access_key=os.getenv("AWS_ACCESS_KEY_ID"),
        secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        secure=False,
    )

    # Nettoyage du dossier local
    if os.path.exists(local_dir):
        shutil.rmtree(local_dir)
    os.makedirs(local_dir, exist_ok=True)

    objects = minio_client.list_objects(bucket_name, recursive=True)
    for obj in objects:
        if obj.object_name.lower().endswith((".jpg", ".png")):
            label = "dandelion" if "dandelion" in obj.object_name.lower() \
                else "grass"
            label_dir = os.path.join(local_dir, label)
            os.makedirs(label_dir, exist_ok=True)
            filename = os.path.basename(obj.object_name)
            file_path = os.path.join(label_dir, filename)

            response = minio_client.get_object(bucket_name, obj.object_name)
            with open(file_path, "wb") as f:
                f.write(response.read())
            response.close()

    print(f"Données copiées dans {local_dir}")
    return local_dir


def main():
    # Configuration MLflow via variables d'environnement
    mlflow.set_tracking_uri(os.getenv("MLFLOW_API"))
    mlflow.set_experiment("classification_dandelion_grass_fastai")

    # Données depuis MinIO
    data_dir = download_minio_dataset()

    # Création DataLoader
    dls = ImageDataLoaders.from_folder(
        data_dir, valid_pct=0.2, item_tfms=Resize(224), bs=32, num_workers=0
    )

    # Modèle
    learn = cnn_learner(dls, resnet34, metrics=accuracy)

    with mlflow.start_run() as run:
        learn.fine_tune(5)

        acc = float(learn.validate()[1])
        preds, targs = learn.get_preds()
        pred_c = preds.argmax(dim=1)

        # Log des paramètres
        mlflow.log_params(
            {
                "architecture": "resnet34",
                "image_size": 224,
                "batch_size": 32,
                "epochs": 5,
            }
        )

        # Log des métriques
        mlflow.log_metrics(
            {
                "accuracy": acc,
                "f1_score": f1_score(targs, pred_c, average="macro"),
                "precision": precision_score(targs, pred_c, average="macro"),
                "recall": recall_score(targs, pred_c, average="macro"),
            }
        )

        # Export et enregistrement du modèle
        model_dir = os.path.abspath("/opt/airflow/scripts/saved_models")
        os.makedirs(model_dir, exist_ok=True)
        export_path = os.path.join(model_dir, "export.pkl")
        learn.export(export_path)
        mlflow.log_artifact(export_path, artifact_path="model")

        # Versionnement dans Model Registry
        client = MlflowClient()  # noqa: F841
        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/model"
        mlflow.register_model(model_uri, "dandelion_classifier")

    print("✅ Modèle entraîné, suivi, exporté et versionné")


if __name__ == "__main__":
    main()
