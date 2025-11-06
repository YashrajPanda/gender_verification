import os
from zipfile import ZipFile

REFERENCE_ZIP = "/content/gender_verification/data/raw/short_references_final.zip"
DISTORTION_ZIP = "/content/gender_verification/data/raw/short_distortion_final.zip"
OUTPUT_DIR = "data/extracted"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("📦 Starting file extraction...")

def extract_zip(src, dest):
    if os.path.exists(src):
        with ZipFile(src, 'r') as zf:
            zf.extractall(dest)
        print(f"✅ Extracted {src} to {dest}")
    else:
        print(f"⚠️ {src} not found!")

extract_zip(REFERENCE_ZIP, f"{OUTPUT_DIR}/ref")
extract_zip(DISTORTION_ZIP, f"{OUTPUT_DIR}/distorted")
