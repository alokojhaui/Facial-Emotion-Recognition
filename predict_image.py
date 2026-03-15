"""
============================================================
  Facial Emotion Recognition — Single Image Prediction
  predict_image.py
============================================================
  Usage:
      python predict_image.py --image path/to/face.jpg
      python predict_image.py --image path/to/photo.jpg --no_detect
============================================================
"""

import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

MODEL_PATH   = 'models/emotion_cnn_best.keras'
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
IMG_SIZE     = 48
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


def predict_from_image(image_path, detect_face=True):
    """
    Load an image, optionally detect a face region,
    predict the emotion, and display a bar chart of probabilities.
    """
    model = load_model(MODEL_PATH)
    img_bgr  = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if detect_face:
        cascade = cv2.CascadeClassifier(CASCADE_PATH)
        faces   = cascade.detectMultiScale(img_gray, scaleFactor=1.1,
                                           minNeighbors=5, minSize=(30,30))
        if len(faces) == 0:
            print("[WARN] No face detected — using entire image.")
            roi = img_gray
        else:
            x, y, w, h = faces[0]
            roi = img_gray[y:y+h, x:x+w]
            # Draw rectangle on display copy
            cv2.rectangle(img_rgb, (x, y), (x+w, y+h), (0, 200, 0), 2)
    else:
        roi = img_gray

    # Pre-process
    face_resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE)).astype('float32') / 255.0
    face_input   = face_resized.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    # Predict
    probs     = model.predict(face_input, verbose=0)[0]
    pred_idx  = np.argmax(probs)
    pred_label = EMOTION_LABELS[pred_idx]
    pred_prob  = probs[pred_idx]

    # ── Visualise ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(img_rgb)
    axes[0].set_title(f"Predicted: {pred_label}  ({pred_prob*100:.1f}%)",
                      fontsize=14, color='green')
    axes[0].axis('off')

    colors = ['tomato' if i == pred_idx else 'steelblue'
              for i in range(len(EMOTION_LABELS))]
    bars = axes[1].barh(EMOTION_LABELS, probs * 100, color=colors)
    axes[1].set_xlabel('Confidence (%)')
    axes[1].set_title('Emotion Probabilities', fontsize=14)
    axes[1].set_xlim(0, 100)
    for bar, p in zip(bars, probs):
        axes[1].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                     f'{p*100:.1f}%', va='center', fontsize=9)

    plt.suptitle('Facial Emotion Recognition', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/prediction_output.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n[RESULT] Emotion : {pred_label}")
    print(f"         Confidence : {pred_prob*100:.2f}%")
    return pred_label, probs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict emotion from image')
    parser.add_argument('--image',    type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--no_detect', action='store_true',
                        help='Skip face detection; use whole image')
    args = parser.parse_args()

    predict_from_image(args.image, detect_face=not args.no_detect)
