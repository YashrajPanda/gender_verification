import os, random
import tensorflow as tf
from tensorflow import keras
from src.utils import *

EPOCHS, BATCH_SIZE, STEPS_PER_EPOCH = 10, 32, 32
MODEL_PATH = "models/face_embedding_model_CLEAN.h5"
DATA_DIR = "data/extracted"

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.makedirs("models", exist_ok=True)

reference, distorted = {}, {}
for root, _, files in os.walk(DATA_DIR):
    for f in files:
        pid = person_id_from_filename(f)
        if pid == "UNKNOWN": continue
        path = os.path.join(root, f)
        if "ref" in root: reference.setdefault(pid, []).append(path)
        else: distorted.setdefault(pid, []).append(path)

PERSON_IDS = list(set(reference.keys()) & set(distorted.keys()))
merged = {k: reference.get(k, []) + distorted.get(k, []) for k in PERSON_IDS}

def pair_generator(paths_by_person, ids, batch_size):
    while True:
        refs, dists, labels = [], [], []
        for _ in range(batch_size):
            pid = random.choice(ids)
            imgs = paths_by_person.get(pid, [])
            if len(imgs) < 2: continue
            if random.random() < 0.5:
                p1, p2 = random.sample(imgs, 2); label = 1.0
            else:
                pid1, pid2 = random.sample(ids, 2)
                p1, p2 = random.choice(paths_by_person[pid1]), random.choice(paths_by_person[pid2])
                label = 0.0
            i1, i2 = load_and_preprocess_image(p1), load_and_preprocess_image(p2)
            if i1 is not None and i2 is not None:
                refs.append(i1); dists.append(i2); labels.append(label)
        yield ((np.array(refs), np.array(dists)), np.array(labels))

input_shape = (IMG_SIZE, IMG_SIZE, 3)

if os.path.exists(MODEL_PATH):
    print("✅ Model found, loading...")
    model = keras.models.load_model(MODEL_PATH, compile=False)
else:
    print("🚀 Training new Siamese model...")
    model = build_siamese_model(input_shape)
    model.compile(optimizer=keras.optimizers.Adam(1e-4), loss=contrastive_loss)
    gen = pair_generator(merged, PERSON_IDS, BATCH_SIZE)
    model.fit(gen, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, verbose=1)
    model.get_layer("feature_extractor").save(MODEL_PATH, include_optimizer=False)
    print("✅ Model saved to:", MODEL_PATH)
  