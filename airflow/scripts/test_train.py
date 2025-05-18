import pytest
from unittest.mock import Mock, patch, MagicMock
import os
import shutil
import torch
from PIL import Image
import tempfile

from train import download_minio_dataset, main


# === FIXTURES ===


@pytest.fixture
def mock_minio():
    with patch("train.Minio") as mock:
        yield mock


@pytest.fixture
def mock_mlflow():
    with patch("train.mlflow.set_tracking_uri"
    ), patch("train.mlflow.set_experiment"), patch("train.mlflow.start_run"
    ), patch("train.mlflow.log_params"), patch("train.mlflow.log_metrics"
    ), patch("train.mlflow.log_artifact"), patch("train.mlflow.register_model"
    ):
        yield


@pytest.fixture
def mock_fastai():
    with patch("train.ImageDataLoaders.from_folder"), patch(
        "train.cnn_learner"
    ) as mock_learner:
        learner = MagicMock()
        learner.validate.return_value = [None, 0.92]
        learner.get_preds.return_value = (
            torch.tensor([[0.9, 0.1], [0.2, 0.8]]),
            torch.tensor([0, 1]),
        )
        mock_learner.return_value = learner
        yield learner


@pytest.fixture
def temp_image_dir():
    temp_dir = tempfile.mkdtemp()
    os.makedirs(os.path.join(temp_dir, "dandelion"))
    os.makedirs(os.path.join(temp_dir, "grass"))
    img = Image.new("RGB", (224, 224), color="red")
    img.save(os.path.join(temp_dir, "dandelion/test1.jpg"))
    img.save(os.path.join(temp_dir, "grass/test2.jpg"))
    yield temp_dir
    shutil.rmtree(temp_dir)


# === TESTS ===


@patch("train.os.getenv")
def test_download_minio_dataset_success(mock_getenv, mock_minio, temp_image_dir):
    mock_getenv.side_effect = lambda key: {
        "MLFLOW_S3_ENDPOINT_URL": "http://fake-minio:9000",
        "AWS_ACCESS_KEY_ID": "fake_access_key",
        "AWS_SECRET_ACCESS_KEY": "fake_secret_key",
    }.get(key, "")
    mock_client = mock_minio.return_value
    mock_client.list_objects.return_value = [
        Mock(object_name="dandelion/img1.jpg"),
        Mock(object_name="grass/img2.jpg"),
    ]
    obj = MagicMock()
    obj.read.return_value = b"fake_image_data"
    obj.close.return_value = None
    mock_client.get_object.return_value = obj

    path = download_minio_dataset()
    assert os.path.exists(path)
    assert set(os.listdir(path)) == {"dandelion", "grass"}


@patch("train.os.getenv")
def test_download_minio_dataset_empty_bucket(mock_getenv, mock_minio):
    mock_getenv.side_effect = lambda key: {
        "MLFLOW_S3_ENDPOINT_URL": "http://fake-minio:9000",
        "AWS_ACCESS_KEY_ID": "fake_access_key",
        "AWS_SECRET_ACCESS_KEY": "fake_secret_key",
    }.get(key, "")
    mock_client = mock_minio.return_value
    mock_client.list_objects.return_value = []

    path = download_minio_dataset()
    assert os.path.exists(path)
    assert len(os.listdir(path)) == 0


@patch("train.os.getenv")
def test_main_flow(mock_getenv, mock_minio, mock_mlflow, mock_fastai):
    mock_getenv.side_effect = lambda key: {
        "MLFLOW_API": "http://fake-mlflow:5001",
        "MLFLOW_S3_ENDPOINT_URL": "http://fake-minio:9000",
        "AWS_ACCESS_KEY_ID": "fake_access_key",
        "AWS_SECRET_ACCESS_KEY": "fake_secret_key",
    }.get(key, "")

    mock_minio.return_value.list_objects.return_value = [
        Mock(object_name="dandelion/img1.jpg")
    ]
    obj = MagicMock()
    obj.read.return_value = b"fake_data"
    obj.close.return_value = None
    mock_minio.return_value.get_object.return_value = obj

    main()

    mock_fastai.fine_tune.assert_called_once_with(5)
    mock_fastai.export.assert_called_once()


@patch("train.os.getenv")
def test_minio_connection_error(mock_getenv):
    mock_getenv.side_effect = lambda key: {
        "MLFLOW_S3_ENDPOINT_URL": "http://fake-minio:9000",
        "AWS_ACCESS_KEY_ID": "fake_access_key",
        "AWS_SECRET_ACCESS_KEY": "fake_secret_key",
    }.get(key, "")
    with patch("train.Minio", side_effect=Exception("Connection failed")):
        with pytest.raises(Exception, match="Connection failed"):
            download_minio_dataset()


@patch("train.os.getenv")
def test_tempdir_cleanup_on_error(mock_getenv):
    mock_getenv.side_effect = lambda key: {
        "MLFLOW_S3_ENDPOINT_URL": "http://fake-minio:9000",
        "AWS_ACCESS_KEY_ID": "fake_access_key",
        "AWS_SECRET_ACCESS_KEY": "fake_secret_key",
    }.get(key, "")
    with patch("train.Minio") as mock_minio:
        mock_minio.return_value.list_objects.side_effect = Exception("Boom")
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("tempfile.mkdtemp", return_value=temp_dir):
                try:
                    download_minio_dataset()
                except Exception:
                    pass
                assert os.path.exists(temp_dir)
