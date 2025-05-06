import requests
import psycopg2
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

# Connexion à PostgreSQL
conn = psycopg2.connect(
    dbname="plants_db",
    user="postgres",
    password="password",
    host="localhost",
    port="5432"
)
cursor = conn.cursor()

# Récupérer quelques URLs d'images
cursor.execute("SELECT url_s3, label FROM plants_data LIMIT 5;")
images = cursor.fetchall()

# Afficher les images et leurs tailles
for url, label in images:
    response = requests.get(url)
    if response.status_code == 200:
        img = Image.open(BytesIO(response.content))
        print(f"Image {label}: Format={img.format}, Taille={img.size}, Mode={img.mode}")

        plt.imshow(img)
        plt.title(label)
        plt.axis("off")
        plt.show()
    else:
        print(f"❌ Impossible de télécharger {url}")

# Fermer la connexion
cursor.close()
conn.close()
