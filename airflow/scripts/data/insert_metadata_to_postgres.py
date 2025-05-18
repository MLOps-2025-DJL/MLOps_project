import psycopg2

# Configuration of the connection to PostgreSQL on Docker
conn = psycopg2.connect(
    dbname="plants_db",
    user="postgres",
    password="postgres",
    host="postgres",
    port="5432"
)
cursor = conn.cursor()

# Create a table if not exist
cursor.execute("""
    CREATE TABLE IF NOT EXISTS plants_data (
        id SERIAL PRIMARY KEY,
        url_source TEXT NOT NULL,
        url_s3 TEXT NOT NULL,
        label TEXT CHECK(label IN ('dandelion', 'grass')) NOT NULL
    );
""")

# Teplates of URLs
source_url_template = "https://raw.githubusercontent.com/btphan95/greenr-airflow/refs/heads/master/data/{label}/{index:08d}.jpg"
s3_url_template = "https://mlops-plants-data.s3.amazonaws.com/{label}/{index:08d}.jpg"

# Labels and index
labels = ["dandelion", "grass"]
num_images = 200  # 200 imágenes por categoría

# Insert data on PostgreSQL
for label in labels:
    for index in range(num_images):
        url_source = source_url_template.format(label=label, index=index)
        url_s3 = s3_url_template.format(label=label, index=index)
        
        # Verificar si ya existe para no duplicar
        cursor.execute(
            "SELECT 1 FROM plants_data WHERE url_s3 = %s LIMIT 1;",
            (url_s3,)
        )
        exists = cursor.fetchone()
        
        if not exists:
            cursor.execute(
                "INSERT INTO plants_data (url_source, url_s3, label) VALUES (%s, %s, %s)",
                (url_source, url_s3, label)
            )

# Save changes and and close conexion
conn.commit()
cursor.close()
conn.close()

print("Completed.")
