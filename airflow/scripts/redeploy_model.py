import requests
import os

# Load the API endpoint URL from environment variables or use the default
API_RELOAD_URL = os.getenv("API_RELOAD_URL", "http://api:8000/reload")

try:
    print(f"Contacting API at {API_RELOAD_URL} to reload model...")
    response = requests.post(API_RELOAD_URL, timeout=5)

    if response.status_code == 200:
        print("API responded with success. Model reloaded.")
    else:
        print(f"API responded with status code {response.status_code}: {response.text}")

except requests.exceptions.ConnectionError:
    print("Connection failed: API is not reachable.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
