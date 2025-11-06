import random, os, numpy as np, matplotlib.pyplot as plt
from tensorflow import keras
from src.utils import *

VERIFICATION_THRESHOLD = 0.6
MODEL_PATH = "models/face_embedding_model_CLEAN.h5"
DATA_DIR = "data/extracted"

model = keras.models.load_model(MODEL_PATH, compile=False)
reference, distorted = {}, {}

for root, _, files in os.walk(DATA_DIR):
    for f in files:
        pid = person_id_from_filename(f)
        if "ref" in root: reference.setdefault(pid, []).append(os.path.join(root,f))
        else: distorted.setdefault(pid, []).append(os.path.join(root,f))

ids = list(set(reference.keys()) & set(distorted.keys()))
pid = random.choice(ids)
ref_path, query_path = reference[pid][0], distorted[pid][0]

ref, query = load_and_preprocess_image(ref_path), load_and_preprocess_image(query_path)
emb_ref, emb_query = model.predict(np.expand_dims(ref,0))[0], model.predict(np.expand_dims(query,0))[0]
dist = np.linalg.norm(emb_ref - emb_query)

verdict = "Same Person ✅" if dist < VERIFICATION_THRESHOLD else "Different Person ❌"
print(f"ID: {pid}\nDistance: {dist:.4f}\nVerdict: {verdict}")

fig, ax = plt.subplots(1,2,figsize=(10,5))
ax[0].imshow(ref); ax[0].set_title("Reference")
ax[1].imshow(query); ax[1].set_title(f"Query\n{verdict}\nDist: {dist:.3f}")
for a in ax: a.axis("off")
plt.show()
