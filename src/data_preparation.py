import os
import cv2
import numpy as np
from zipfile import ZipFile
from src.config import DATA_DIR, IMG_SIZE

def extract_zip(zip_path, extract_to):
    """Extracts a given zip file."""
    os.makedirs(extract_to, exist_ok=True)
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"✅ Extracted: {zip_path}")

def load_and_preprocess_image(path):
    """Loads, resizes, and normalizes an image."""
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype('float32') / 255.0
    return img
