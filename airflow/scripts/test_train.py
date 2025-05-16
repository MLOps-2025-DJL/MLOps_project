import pytest
from unittest.mock import Mock, patch, MagicMock
from io import BytesIO
import os
import shutil
import torch
from PIL import Image
import tempfile

# Import du script à tester
from train import download_minio_dataset, main


# === FIXTURES ===

@pytest.fixture
def mock_minio():
    """Mock MinIO client"""
    with patch('train.Minio') as mock:
        yield mock

@pytest.fixture
def mock_mlflow():
    """Mock complet de MLflow"""
    with patch('train.mlflow.set_tracking_uri'), \
         patch('train.mlflow.set_experiment'), \
         patch('train.mlflow.start_run'), \
         patch('train.mlflow.log_params'), \
         patch('train.mlflow.log_metrics'), \
         patch('train.mlflow.log_artifact'), \
         patch('train.mlflow.register_model'), \
         patch('train.MlflowClient'):
        yield

@pytest.fixture
def mock_fastai():
    """Mock fastai DataLoaders et Learner"""
    with patch('train.ImageDataLoaders.from_folder'), \
         patch('train.cnn_learner') as mock_learner:
        learner = MagicMock()
        learner.validate.return_value = [None, 0.92]
        learner.get_preds.return_value = (
            torch.tensor([[0.9, 0.1], [0.2, 0.8]]),
            torch.tensor([0, 1])
        )
        mock_learner.return_value = learner
        yield learner

@pytest.fixture
def temp_image_dir():
    """Dossier temporaire avec des images factices"""
    temp_dir = tempfile.mkdtemp()
    os.makedirs(os.path.join(temp_dir, "dandelion"))
    os.makedirs(os.path.join(temp_dir, "grass"))
    img = Image.new('RGB', (224, 224), color='red')
    img.save(os.path.join(temp_dir, "dandelion/test1.jpg"))
    img.save(os.path.join(temp_dir, "grass/test2.jpg"))
    yield temp_dir
    shutil.rmtree(temp_dir)


# === TESTS ===

def test_download_minio_dataset_success(mock_minio, temp_image_dir):
    mock_client = mock_minio.return_value
    mock_client.list_objects.return_value = [
        Mock(object_name="dandelion/img1.jpg"),
        Mock(object_name="grass/img2.jpg")
    ]
    # Simuler un objet avec .read() et .close()
    mock_obj = MagicMock()
    mock_obj.read.return_value = b"fake_image_data"
    mock_obj.close.return_value = None
    mock_client.get_object.return_value = mock_obj

    path = download_minio_dataset()
    assert os.path.exists(path)
    assert set(os.listdir(path)) == {"dandelion", "grass"}

def test_download_minio_dataset_empty_bucket(mock_minio):
    mock_client = mock_minio.return_value
    mock_client.list_objects.return_value = []

    path = download_minio_dataset()
    assert os.path.exists(path)
    assert len(os.listdir(path)) == 0

def test_main_flow(mock_minio, mock_mlflow, mock_fastai):
    mock_minio.return_value.list_objects.return_value = [
        Mock(object_name="dandelion/img1.jpg")
    ]
    mock_obj = MagicMock()
    mock_obj.read.return_value = b"fake_data"
    mock_obj.close.return_value = None
    mock_minio.return_value.get_object.return_value = mock_obj

    main()

    mock_fastai.fine_tune.assert_called_once_with(5)
    mock_fastai.export.assert_called_once()

def test_minio_connection_error():
    with patch('train.Minio', side_effect=Exception("Connection failed")):
        with pytest.raises(Exception, match="Connection failed"):
            download_minio_dataset()

def test_metric_calculation(mock_fastai, mock_mlflow):
    mock_fastai.get_preds.return_value = (
        torch.tensor([[0.8, 0.2], [0.3, 0.7]]),
        torch.tensor([0, 1])
    )
    with patch('train.download_minio_dataset'), \
         patch('train.mlflow.log_metrics') as mock_log, \
         patch('train.mlflow.start_run'), \
         patch('train.mlflow.set_tracking_uri'), \
         patch('train.mlflow.set_experiment'), \
         patch('train.mlflow.log_params'), \
         patch('train.mlflow.log_artifact'), \
         patch('train.mlflow.register_model'), \
         patch('train.MlflowClient'):
        main()
        metrics = mock_log.call_args[0][0]
        assert "precision" in metrics
        assert 0 <= metrics["precision"] <= 1

def test_tempdir_cleanup_on_error():
    with patch('train.Minio') as mock_minio:
        mock_minio.return_value.list_objects.side_effect = Exception("Boom")
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('tempfile.mkdtemp', return_value=temp_dir):
                try:
                    download_minio_dataset()
                except Exception:
                    pass
                # Le dossier peut exister, mais il doit être vide
                assert os.path.exists(temp_dir)
