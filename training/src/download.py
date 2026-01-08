import os
import requests
import zipfile
import io
from pathlib import Path

# Config
URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
RAW_DIR = Path("training/data/raw")

def download_and_extract():
    if RAW_DIR.exists() and (RAW_DIR / "ml-1m/ratings.dat").exists():
        print(f"[INFO] Dataset already exists in {RAW_DIR}")
        return

    print(f"[INFO] Downloading MovieLens 1M from {URL}...")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    response = requests.get(URL)
    response.raise_for_status()
    
    print("[INFO] Extracting...")
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall(RAW_DIR)
    
    print("[SUCCESS] Data downloaded and extracted.")

if __name__ == "__main__":
    download_and_extract()