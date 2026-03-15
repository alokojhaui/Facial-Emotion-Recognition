"""
============================================================
  Facial Emotion Recognition — Real-Time Webcam Detection
  realtime_detect.py
============================================================
  Uses OpenCV to:
    1. Capture webcam frames
    2. Detect faces with Haar cascade
    3. Predict emotion with the trained CNN
    4. Overlay label + probability bar on the frame
============================================================
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_PATH    = 'models/emotion_cnn_best.keras'
CASCADE_PATH  = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
IMG_SIZE      = 48
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Distinct colours per emotion (BGR)
EMOTION_COLORS = {
    'Angry':    (0,   0,   220),
    'Disgust':  (0,   140, 0),
    'Fear':     (128, 0,   128),
    'Happy':    (0,   200, 255),
    'Sad':      (180, 100, 0),
    'Surprise': (0,   165, 255),
    'Neutral':  (180, 180, 180),
}


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def preprocess_face(face_gray):
    """Resize → normalise → add batch & channel dims."""
    face = cv2.resize(face_gray, (IMG_SIZE, IMG_SIZE))
    face = face.astype('float32') / 255.0
    face = np.expand_dims(face, axis=-1)   # (48,48,1)
    face = np.expand_dims(face, axis=0)    # (1,48,48,1)
    return face


def draw_emotion_bars(frame, x, y, w, h, probs):
    """Draw a small probability bar chart beside the face box."""
    bar_x = x + w + 10
    bar_max_w = 120
    bar_h = 14
    bar_gap = 4
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, (label, prob) in enumerate(zip(EMOTION_LABELS, probs)):
        bar_y = y + i * (bar_h + bar_gap)
        color = EMOTION_COLORS[label]

        # Background track
        cv2.rectangle(frame,
                      (bar_x, bar_y),
                      (bar_x + bar_max_w, bar_y + bar_h),
                      (60, 60, 60), -1)

        # Filled portion
        filled = int(prob * bar_max_w)
        cv2.rectangle(frame,
                      (bar_x, bar_y),
                      (bar_x + filled, bar_y + bar_h),
                      color, -1)

        # Label + percentage text
        cv2.putText(frame,
                    f"{label[:3]} {prob*100:.0f}%",
                    (bar_x + bar_max_w + 4, bar_y + bar_h - 2),
                    font, 0.38, (230, 230, 230), 1, cv2.LINE_AA)


def run_realtime(camera_index=1):
    """
    Main loop:
      • Read frame from webcam
      • Convert to grayscale
      • Detect faces
      • Predict emotion for each face
      • Annotate and display frame
    Press 'q' to quit, 's' to save a screenshot.
    """
    print("[INFO] Loading model ...")
    model = load_model(MODEL_PATH)
    print("[INFO] Model loaded.")

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Check camera index.")
        return

    print("[INFO] Starting webcam ... Press 'q' to quit, 's' to screenshot.")
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Frame capture failed.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── Face detection ─────────────────────────────────────────
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            face_input = preprocess_face(roi_gray)

            # ── Prediction ─────────────────────────────────────────
            probs  = model.predict(face_input, verbose=0)[0]
            pred_idx   = np.argmax(probs)
            pred_label = EMOTION_LABELS[pred_idx]
            pred_prob  = probs[pred_idx]
            color = EMOTION_COLORS[pred_label]

            # ── Draw face box ──────────────────────────────────────
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            # ── Main label above box ───────────────────────────────
            label_text = f"{pred_label}  {pred_prob*100:.1f}%"
            (tw, th), _ = cv2.getTextSize(label_text,
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame,
                          (x, y - th - 10),
                          (x + tw + 6, y),
                          color, -1)
            cv2.putText(frame, label_text,
                        (x + 3, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)

            # ── Probability bars ───────────────────────────────────
            draw_emotion_bars(frame, x, y, w, h, probs)

        # ── HUD overlay ────────────────────────────────────────────
        cv2.putText(frame,
                    f"Faces detected: {len(faces)}",
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (100, 255, 100), 2, cv2.LINE_AA)
        cv2.putText(frame,
                    "Press 'q' quit | 's' screenshot",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow('Facial Emotion Recognition — Real-Time', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            screenshot_path = f'results/screenshot_{frame_count:04d}.png'
            cv2.imwrite(screenshot_path, frame)
            print(f"[INFO] Screenshot saved → {screenshot_path}")
            frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam released. Goodbye!")


if __name__ == '__main__':
    run_realtime(camera_index=1)
