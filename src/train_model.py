import os
import random
import numpy as np
import tensorflow as tf
from src.config import IMG_SIZE, EPOCHS, STEPS_PER_EPOCH, BATCH_SIZE, MODEL_PATH
from src.data_preparation import load_and_preprocess_image
from src.model_siamese import build_siamese_model, contrastive_loss

def siamese_pair_generator(image_paths_by_person, person_ids, batch_size):
    while True:
        ref_images, dist_images, labels = [], [], []
        for _ in range(batch_size):
            pid = random.choice(person_ids)
            paths = image_paths_by_person.get(pid, [])
            if len(paths) < 2:
                continue
            if random.random() < 0.5:
                img1, img2 = random.sample(paths, 2)
                label = 1.0
            else:
                p1, p2 = random.sample(person_ids, 2)
                img1 = random.choice(image_paths_by_person[p1])
                img2 = random.choice(image_paths_by_person[p2])
                label = 0.0
            im1 = load_and_preprocess_image(img1)
            im2 = load_and_preprocess_image(img2)
            if im1 is not None and im2 is not None:
                ref_images.append(im1)
                dist_images.append(im2)
                labels.append(label)
        yield ((np.array(ref_images), np.array(dist_images)), np.array(labels))

def train_model(image_paths_by_person, person_ids):
    input_shape = (IMG_SIZE, IMG_SIZE, 3)
    siamese_model = build_siamese_model(input_shape)
    siamese_model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=contrastive_loss)
    generator = siamese_pair_generator(image_paths_by_person, person_ids, BATCH_SIZE)
    siamese_model.fit(generator, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH)
    feature_extractor = siamese_model.get_layer("feature_extractor")
    os.makedirs("models", exist_ok=True)
    feature_extractor.save(MODEL_PATH, include_optimizer=False)
    print(f"✅ Model saved at {MODEL_PATH}")
    return feature_extractor
