from deepface import DeepFace
import tensorflow as tf

def detect_gender(image_path):
    """Performs gender detection using DeepFace."""
    tf.keras.backend.clear_session()
    try:
        result = DeepFace.analyze(img_path=image_path, actions=['gender'], enforce_detection=False)
        gender = result[0]['dominant_gender']
        confidence = result[0]['gender'][gender]
    except Exception as e:
        print(f"⚠️ Gender detection failed for {image_path}: {e}")
        gender, confidence = "Unknown", 0.0
    return gender, confidence
