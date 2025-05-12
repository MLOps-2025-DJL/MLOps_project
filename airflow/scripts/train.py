from fastai.vision.all import *
import os
import mlflow

def main():
    # ENV pour MinIO
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000"
    os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"

    # MLflow
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("classification_dandelion_grass_fastai")

    # Données
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "data"))
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dossier introuvable : {data_dir}")

    # DataLoader avec num_workers=0 (Windows safe)
    dls = ImageDataLoaders.from_folder(
        data_dir,
        valid_pct=0.2,
        item_tfms=Resize(224),
        bs=32,
        num_workers=0  # 👈 important sur Windows
    )

    # Modèle
    learn = cnn_learner(dls, resnet34, metrics=accuracy)

    with mlflow.start_run():
        learn.fine_tune(5)

        acc = float(learn.validate()[1])
        mlflow.log_param("architecture", "resnet34")
        mlflow.log_param("image_size", 224)
        mlflow.log_param("batch_size", 32)
        mlflow.log_param("epochs", 5)
        mlflow.log_metric("accuracy", acc)

        model_dir = os.path.abspath("saved_models")
        os.makedirs(model_dir, exist_ok=True)
        export_path = os.path.join(model_dir, "export.pkl")
        learn.export(export_path)

        mlflow.log_artifact(export_path, artifact_path="model")

    print("✅ Modèle entraîné, loggé MLflow + exporté MinIO")

# 💡 Protection Windows pour multiprocessing
if __name__ == "__main__":
    main()
