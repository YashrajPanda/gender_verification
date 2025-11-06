import cv2, os, matplotlib.pyplot as plt, tensorflow as tf
from deepface import DeepFace

def detect_gender(img_path):
    try:
        tf.keras.backend.clear_session()
        res = DeepFace.analyze(img_path=img_path, actions=['gender'], enforce_detection=False)
        g = res[0]['dominant_gender']; conf = res[0]['gender'][g]
        return g, conf
    except Exception as e:
        print("⚠️ Gender detection failed:", e); return "Unknown", 0.0

if __name__ == "__main__":
    ref_img = "data/extracted/ref/any_image.jpg"  # change as needed
    gender, conf = detect_gender(ref_img)
    print(f"Gender: {gender}, Confidence: {conf:.1f}%")
