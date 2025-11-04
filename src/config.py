import os

# Image and training parameters
IMG_SIZE = 128
EMBEDDING_DIM = 64
MARGIN = 1.0
EPOCHS = 10
BATCH_SIZE = 32
STEPS_PER_EPOCH = 32
VERIFICATION_THRESHOLD = 0.6

# File paths
DATA_DIR = "./data/extracted"
MODEL_PATH = "./models/face_embedding_model_CLEAN.h5"

# Cascade file for face detection
CASCADE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
CASCADE_FILE = "haarcascade_frontalface_default.xml"
