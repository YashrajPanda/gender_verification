# 👥 Gender Detection & Face Verification using Siamese CNN

This project combines **Siamese Neural Networks** for face verification and **DeepFace** for gender classification.

## 🚀 Features
- Extracts and preprocesses facial image pairs.
- Trains a Siamese CNN with contrastive loss.
- Performs gender detection using DeepFace.
- Provides visualization and verification of pairs.

## 📂 Folder Structure
See the `src/` directory for modular Python scripts.

## 🧠 Model Overview
- Siamese CNN uses shared weights for feature embeddings.
- Euclidean distance measures similarity between faces.

## ⚙️ Requirements
Install all dependencies:
```bash
pip install -r requirements.txt

py -3.12 -m venv
