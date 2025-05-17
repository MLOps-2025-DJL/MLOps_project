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

1.⁠ ⁠Construire l'image Docker de base :
docker build -f Dockerfile.base -t base .
2.⁠ ⁠Démarrer les services avec Docker Compose
docker compose up -d
3.⁠ ⁠Lancer le DAG dans Airflow
Ouvre l’interface web d’Airflow (accessible à l’adresse http://localhost:8088). Lance le DAG