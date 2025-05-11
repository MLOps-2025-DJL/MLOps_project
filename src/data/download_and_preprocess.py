import os
import requests
from PIL import Image
from io import BytesIO
import psycopg2

# Database configuration
conn = psycopg2.connect(
    dbname="plants_db",
    user="postgres",
    password="password",
    host="localhost",
    port="5432"
)
cursor = conn.cursor()

# SQL query to get URLs
cursor.execute("SELECT url_source, label FROM plants_data")
rows = cursor.fetchall()

# Local save path
SAVE_PATH = "MLOps_project/data/preprocess"
IMAGE_SIZE = (256, 256)

# Create directories
os.makedirs(f"{SAVE_PATH}/dandelion", exist_ok=True)
os.makedirs(f"{SAVE_PATH}/grass", exist_ok=True)

# Loop through each row and download the image
for url, label in rows:
    print(f"\n Downloading and preprocessing images for '{label}'")
    img_name = url.split("/")[-1]  # Get the filename from the URL
    local_path = f"{SAVE_PATH}/{label}/{img_name}"

    try:
        # Download image
        response = requests.get(url)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content)).convert('RGB')
            image = image.resize(IMAGE_SIZE)
            image.save(local_path)
            print(f"Saved: {local_path}")
        else:
            print(f"Could not download: {url}")
    except Exception as e:
        print(f"Error processing {url}: {e}")

# Close the connection
cursor.close()
conn.close()
