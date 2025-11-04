import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from src.config import EMBEDDING_DIM, MARGIN

def build_feature_extractor(input_shape):
    """Feature extractor CNN."""
    inp = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, (3, 3), activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    out = layers.Dense(EMBEDDING_DIM)(x)
    return keras.Model(inp, out, name="feature_extractor")

def build_siamese_model(input_shape):
    """Creates Siamese model using shared feature extractor."""
    feature_extractor = build_feature_extractor(input_shape)
    input_A = keras.Input(shape=input_shape)
    input_B = keras.Input(shape=input_shape)
    emb_A = feature_extractor(input_A)
    emb_B = feature_extractor(input_B)
    distance = layers.Lambda(lambda x: tf.sqrt(tf.reduce_sum(tf.square(x[0] - x[1]), axis=-1, keepdims=True)))(
        [emb_A, emb_B])
    return keras.Model([input_A, input_B], distance, name="siamese_network")

def contrastive_loss(y_true, y_pred, margin=MARGIN):
    """Custom Contrastive Loss."""
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(tf.cast(y_true, tf.float32) * square_pred +
                          (1 - tf.cast(y_true, tf.float32)) * margin_square)
