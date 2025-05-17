import unittest
import sys
from unittest.mock import patch, MagicMock, mock_open
import io
import os
import pytest
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np
import boto3
from datetime import datetime

# Import le module à tester
from main import app, find_latest_model, load_model, predict_image

# Créer un client de test
client = TestClient(app)

class TestMainPy(unittest.TestCase):
    
    def setUp(self):
        # Configurer tous les mocks nécessaires pour les tests
        self.patcher_s3 = patch('boto3.client')
        self.mock_s3 = self.patcher_s3.start()
        
        # Mock pour load_learner
        self.patcher_load_learner = patch('main.load_learner')
        self.mock_load_learner = self.patcher_load_learner.start()
        
        # Mock pour Image.open
        self.patcher_image = patch('PIL.Image.open')
        self.mock_image = self.patcher_image.start()
        
    def tearDown(self):
        # Arrêter tous les patchers
        self.patcher_s3.stop()
        self.patcher_load_learner.stop()
        self.patcher_image.stop()
    
    def test_find_latest_model(self):
        # Configurer le mock pour boto3.client
        mock_s3_client = MagicMock()
        self.mock_s3.return_value = mock_s3_client
        
        # Simuler les buckets
        mock_s3_client.list_buckets.return_value = {'Buckets': [{'Name': 'test-bucket'}]}
        
        # Simuler les objets dans le bucket
        test_date = datetime(2023, 1, 1)
        mock_s3_client.list_objects_v2.return_value = {
            'Contents': [
                {'Key': 'model_v1.pkl', 'LastModified': datetime(2023, 1, 1)},
                {'Key': 'model_v2.pkl', 'LastModified': datetime(2023, 2, 1)},
                {'Key': 'data.csv', 'LastModified': datetime(2023, 3, 1)}
            ]
        }
        
        # Appeler la fonction à tester
        result = find_latest_model()
        
        # Vérifier les résultats
        self.assertEqual(result['Bucket'], 'test-bucket')
        self.assertEqual(result['Key'], 'model_v2.pkl')
        self.assertEqual(result['LastModified'], datetime(2023, 2, 1))
        
    def test_find_latest_model_no_pkl(self):
        # Configurer le mock pour boto3.client
        mock_s3_client = MagicMock()
        self.mock_s3.return_value = mock_s3_client
        
        # Simuler les buckets
        mock_s3_client.list_buckets.return_value = {'Buckets': [{'Name': 'test-bucket'}]}
        
        # Simuler aucun fichier .pkl
        mock_s3_client.list_objects_v2.return_value = {
            'Contents': [
                {'Key': 'data.csv', 'LastModified': datetime(2023, 3, 1)}
            ]
        }
        
        # Vérifier que la fonction lève une exception
        with self.assertRaises(FileNotFoundError):
            find_latest_model()
    
    def test_load_model(self):
        # Mock pour find_latest_model
        with patch('main.find_latest_model') as mock_find_latest:
            mock_find_latest.return_value = {
                'Bucket': 'test-bucket',
                'Key': 'model.pkl',
                'LastModified': datetime(2023, 1, 1)
            }
            
            # Mock pour boto3.client
            mock_s3_client = MagicMock()
            self.mock_s3.return_value = mock_s3_client
            
            # Simuler la réponse de get_object
            mock_s3_client.get_object.return_value = {
                'Body': MagicMock(read=lambda: b'mock_model_data')
            }
            
            # Mock pour load_learner
            mock_model = MagicMock()
            self.mock_load_learner.return_value = mock_model
            
            # Appeler la fonction à tester
            result = load_model()
            
            # Vérifier les résultats
            self.assertEqual(result, mock_model)
            mock_s3_client.get_object.assert_called_once_with(
                Bucket='test-bucket', 
                Key='model.pkl'
            )
    
    def test_predict_image(self):
        # Créer un mock pour le modèle
        mock_model = MagicMock()
        mock_model.predict.return_value = ('cat', None, MagicMock(max=lambda: MagicMock(item=lambda: 0.95)))
        
        # Créer une image fictive
        test_image = MagicMock()
        
        # Appeler la fonction à tester
        class_name, confidence = predict_image(test_image, mock_model)
        
        # Vérifier les résultats
        self.assertEqual(class_name, 'cat')
        self.assertEqual(confidence, 0.95)
        mock_model.predict.assert_called_once_with(test_image)
    
    @patch('main.predict_image')
    def test_predict_endpoint_success(self, mock_predict):
        # Configurer les mocks
        mock_predict.return_value = ('cat', 0.95)
        
        # Simuler un modèle chargé
        with patch('main.model', MagicMock()):
            # Créer une image de test
            test_image = Image.new('RGB', (100, 100))
            img_byte_arr = io.BytesIO()
            test_image.save(img_byte_arr, format='JPEG')
            img_byte_arr.seek(0)
            
            # Faire une requête à l'endpoint
            response = client.post(
                "/predict",
                files={"file": ("test_image.jpg", img_byte_arr, "image/jpeg")}
            )
            
            # Vérifier la réponse
            self.assertEqual(response.status_code, 200)
            json_response = response.json()
            self.assertEqual(json_response["prediction"], "cat")
            self.assertEqual(json_response["probability"], 0.95)
    
    def test_predict_endpoint_no_model(self):
        # Simuler l'absence de modèle et une erreur lors du chargement
        with patch('main.model', None), \
             patch('main.load_model', side_effect=Exception("Test error")):
            
            # Créer une image de test
            test_image = Image.new('RGB', (100, 100))
            img_byte_arr = io.BytesIO()
            test_image.save(img_byte_arr, format='JPEG')
            img_byte_arr.seek(0)
            
            # Faire une requête à l'endpoint
            response = client.post(
                "/predict",
                files={"file": ("test_image.jpg", img_byte_arr, "image/jpeg")}
            )
            
            # Vérifier la réponse d'erreur
            self.assertEqual(response.status_code, 500)
            json_response = response.json()
            self.assertIn("Could not load model", json_response["detail"])
    
    def test_predict_endpoint_invalid_image(self):
        # Simuler un modèle chargé
        with patch('main.model', MagicMock()), \
            patch('PIL.Image.open', side_effect=IOError("cannot identify image file")):
            # Créer un fichier invalide comme image
            invalid_data = io.BytesIO(b"not an image")
            
            # Faire une requête à l'endpoint
            response = client.post(
                "/predict",
                files={"file": ("invalid.jpg", invalid_data, "image/jpeg")}
            )
            
            # Vérifier la réponse d'erreur
            self.assertEqual(response.status_code, 400)
            json_response = response.json()
            self.assertEqual(json_response["detail"], "Invalid image file")
            
    def test_predict_endpoint_unexpected_model_output(self):
        # Create a valid dummy image
        test_image = Image.new('RGB', (100, 100))
        img_byte_arr = io.BytesIO()
        test_image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)

        # Patch image opening to return the real image
        with patch('PIL.Image.open', return_value=test_image), \
            patch('main.model') as mock_model:

            # Mock model.predict to return an unexpected result (e.g., an empty tuple)
            mock_model.predict.return_value = ()

            # Call the endpoint
            response = client.post(
                "/predict",
                files={"file": ("test.jpg", img_byte_arr, "image/jpeg")}
            )

            # Assert the correct error was raised and handled
            assert response.status_code == 400
            assert "Unexpected prediction output" in response.json()["detail"]

if __name__ == "__main__":
    unittest.main()
