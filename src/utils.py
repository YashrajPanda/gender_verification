import os, cv2, numpy as np, hashlib, random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

IMG_SIZE = 128
EMBEDDING_DIM = 64
MARGIN = 1.0

def person_id_from_filename(filename):
    base_name = os.path.basename(filename)
    if '__' in base_name: return base_name.split('__')[0]
    elif '.' in base_name: return base_name.split('.')[0]
    return "UNKNOWN"

def load_and_preprocess_image(path):
    img = cv2.imread(path)
    if img is None: return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img.astype('float32') / 255.0

def contrastive_loss(y_true, y_pred, margin=MARGIN):
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    loss = tf.cast(y_true, tf.float32) * square_pred + (1 - tf.cast(y_true, tf.float32)) * margin_square
    return tf.reduce_mean(loss)

def build_feature_extractor(input_shape):
    inp = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, (3,3), activation='relu')(inp)
    x = layers.BatchNormalization()(x); x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (3,3), activation='relu')(x)
    x = layers.BatchNormalization()(x); x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, (3,3), activation='relu')(x)
    x = layers.BatchNormalization()(x); x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x); x = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(EMBEDDING_DIM)(x)
    return keras.Model(inp, out, name="feature_extractor")

def build_siamese_model(input_shape):
    feat = build_feature_extractor(input_shape)
    a, b = keras.Input(shape=input_shape), keras.Input(shape=input_shape)
    emb_a, emb_b = feat(a), feat(b)
    dist = layers.Lambda(lambda x: tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(x[0]-x[1]), axis=-1, keepdims=True), 1e-6)),
                         name="euclidean_distance")([emb_a, emb_b])
    return keras.Model([a,b], dist, name="Siamese_Verification_Model")
