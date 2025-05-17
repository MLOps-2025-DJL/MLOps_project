# MLOps_project

Projet MLOPS sur la classification d'images 
Telecom Paris - 2025

Stacks utilisées :
- Microk8S pour le cluster kubernetes
- Airflow pour la pipeline
- Gradio pour webapp
- Github Action pour CI/CD
- Minio en guise de AWS S3
- Postgres pour bdd
- Elasticsearch & Kibana pour le monitoring
- FastAI pour Deep learning frameworks 
- MLflow pour ML Metadata Store 
- FastAPI pour API
- Docker pour conteneurisation 

1. Configuration
copier le fihcier .env.example dans votre dossier en local et remplir les variables

2. Lancement avec les images Docker disponibles sur Docker Hub 
2.1 Utilisez le fichier docker-compose.prod.yml à laide de la commande : docker compose -f docker-compose.prod.yml up -d  
2.2 Lancez le DAG dans l’interface Airflow : http://localhost:8088  
2.3 Accédez aux différents services :  
    Webapp Gradio : http://localhost:7860  
    API FastAPI : http://localhost:8000/docs  
    MLflow : http://localhost:5001  
    MinIO : http://localhost:9003  
    Airflow : http://localhost:8088  

3. Fonctionnement en local en construisant les images docker   
3.1⁠ ⁠Construire l'image Docker de base à l'aide la commande : docker build -f Dockerfile.base -t base .  
3.2⁠ ⁠Démarrer les services avec Docker Compose : docker compose up -d  
3.⁠ ⁠Lancer le DAG dans Airflow  
Ouvre l’interface web d’Airflow (accessible à l’adresse http://localhost:8088). Lancee le DAG et accéder aux services indiqués plus haut  


