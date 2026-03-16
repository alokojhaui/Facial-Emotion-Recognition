
# 🎭 Facial Emotion Recognition using CNN
**A complete beginner-to-intermediate mini-project**
*Python · TensorFlow/Keras · OpenCV · FER2013*

---

## 📁 Project Structure

```
facial_emotion_recognition/
│
├── data/
│   └── fer2013.csv            ← Download from Kaggle (instructions below)
│
├── models/
│   └── emotion_cnn_best.keras ← Saved after training
│
├── results/
│   ├── training_history.png   ← Accuracy & Loss curves
│   ├── confusion_matrix.png   ← Evaluation heatmap
│   ├── sample_predictions.png ← Visual test results
│   ├── class_distribution.png ← Dataset balance chart
│   └── gradcam_example.png    ← Grad-CAM attention maps
│
├── utils/
│   └── helpers.py             ← Shared utility functions
│
├── notebooks/
│   └── EDA_and_Model_Demo.py  ← Interactive exploration guide
│
├── train.py                   ← Main training script ⭐
├── realtime_detect.py         ← Webcam real-time detection ⭐
├── predict_image.py           ← Single image prediction ⭐
├── requirements.txt           ← Python dependencies
└── README.md                  ← This file
```

---

## 📖 Project Overview

Facial Emotion Recognition (FER) is the task of automatically identifying human
emotional states from facial expressions. This project builds a deep Convolutional
Neural Network (CNN) that classifies faces into **7 universal emotion categories**:

| Label | Index |
|-------|-------|
| Angry | 0 |
| Disgust | 1 |
| Fear | 2 |
| Happy | 3 |
| Sad | 4 |
| Surprise | 5 |
| Neutral | 6 |

---

## 📦 Dataset — FER2013

**Source:** Kaggle — [FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)

**Properties:**
- **48 × 48** grayscale face images
- ~**35,887** labelled samples
- Pre-split into Training / PublicTest / PrivateTest

> **Note on class imbalance:** The *Disgust* class has only ~600 samples vs ~8,900 for *Happy*.
> The model handles this via data augmentation and dropout regularisation.

### Download Instructions

```bash
# Option 1 — Kaggle CLI
pip install kaggle
kaggle datasets download -d msambare/fer2013
unzip fer2013.zip -d data/

# Option 2 — Manual
# Visit https://www.kaggle.com/datasets/msambare/fer2013
# Download fer2013.csv and place it in the data/ folder
```

---

## ⚙️ Installation

```bash
# 1. Clone / download this project
cd facial_emotion_recognition

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate.bat       # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Step 1 — Train the Model
```bash
python train.py
```
This will:
- Load and preprocess FER2013
- Apply data augmentation
- Train the CNN for up to 60 epochs with early stopping
- Save the best model to `models/emotion_cnn_best.keras`
- Output accuracy/loss plots and confusion matrix to `results/`

Expected test accuracy: **~65–68%** (state-of-the-art on FER2013 is ~73%)

---

### Step 2 — Predict from a Single Image
```bash
python predict_image.py --image path/to/face.jpg
```
Optional flag: `--no_detect` skips face detection and uses the whole image.

---

### Step 3 — Real-Time Webcam Detection
```bash
python realtime_detect.py
```
- Press **`q`** to quit
- Press **`s`** to save a screenshot

---

## 🧠 CNN Architecture

```
Input (48×48×1)
  │
  ├─ Conv Block 1: Conv2D(64) → BN → Conv2D(64) → BN → MaxPool → Dropout(0.25)
  ├─ Conv Block 2: Conv2D(128) → BN → Conv2D(128) → BN → MaxPool → Dropout(0.25)
  ├─ Conv Block 3: Conv2D(256) → BN → Conv2D(256) → BN → MaxPool → Dropout(0.25)
  ├─ Conv Block 4: Conv2D(512) → BN → Conv2D(512) → BN → MaxPool → Dropout(0.25)
  │
  ├─ GlobalAveragePooling2D
  ├─ Dense(512) → BN → Dropout(0.5)
  ├─ Dense(256) → Dropout(0.3)
  └─ Dense(7, softmax)
```

**Why these design choices?**
- **Batch Normalisation** — stabilises training, speeds convergence
- **Dropout** — primary regulariser to fight overfitting (25% after conv blocks, 50% after first dense)
- **GlobalAveragePooling** — fewer parameters than Flatten; acts as implicit regulariser
- **L2 Regularisation** — penalises large weights across all conv layers
- **Adam optimiser** with ReduceLROnPlateau — adaptive learning rate for stable convergence

---

## 📊 Training Details

| Hyperparameter | Value |
|----------------|-------|
| Image Size | 48 × 48 |
| Batch Size | 64 |
| Max Epochs | 60 |
| Initial LR | 0.001 |
| Early Stopping Patience | 12 epochs |
| LR Reduction Factor | 0.5 (patience=5) |

**Data Augmentation applied:**
- Random rotation ±15°
- Horizontal flip
- Width/height shift ±10%
- Shear + zoom ±10%

---

## 📈 Results

| Metric | Value |
|--------|-------|
| Train Accuracy | ~74–78% |
| Validation Accuracy | ~64–66% |
| Test Accuracy | ~64–68% |

> These are typical values. FER2013 is a notoriously difficult dataset due to label noise,
> ambiguous expressions, and class imbalance.

---

## 🔭 Possible Improvements

1. **Transfer Learning** — Use pretrained ResNet50 / EfficientNet (fine-tuned on FER2013)
2. **Class Weights** — Apply `class_weight` parameter to handle Disgust imbalance
3. **Focal Loss** — Down-weights easy examples, focuses on hard misclassifications
4. **Attention Mechanisms** — Squeeze-and-Excitation blocks to focus on key facial regions
5. **Ensemble Models** — Average predictions from multiple CNNs
6. **Multi-task Learning** — Jointly predict AU (Action Units) + emotion
7. **Better Dataset** — AffectNet (450K images) or RAF-DB for higher accuracy
8. **Face Alignment** — Align faces by eye position before classification
9. **Temporal Modelling** — Use LSTM or Transformer over frame sequences for video

---

## 🌐 Real-World Applications

| Domain | Use Case |
|--------|----------|
| Healthcare | Patient pain/distress monitoring, mental health screening |
| Education | Student engagement & attention tracking in e-learning |
| Automotive | Driver fatigue / drowsiness detection |
| Retail | Customer sentiment analysis at checkout / in-store |
| Entertainment | Adaptive games / media that responds to player emotion |
| HR & Interviews | Candidate stress/confidence analysis |
| Security | Suspicious behaviour detection in CCTV |
| Accessibility | Emotion-based communication aid for non-verbal individuals |

---

## ⚠️ Ethical Considerations

- Always obtain **informed consent** before capturing/analysing facial data
- Be aware of **demographic bias** — models perform worse on under-represented groups
- Emotion recognition results should be **probabilistic hints**, not definitive conclusions
- Comply with **GDPR / local privacy regulations** in any deployment

---

## 📚 References

1. Goodfellow et al. (2013) — *Challenges in Representation Learning: A report on three machine learning contests* (FER2013 origin paper)
2. LeCun et al. (1998) — *Gradient-Based Learning Applied to Document Recognition* (CNN fundamentals)
3. Selvaraju et al. (2017) — *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*
4. TensorFlow/Keras Documentation — https://keras.io
5. OpenCV Documentation — https://docs.opencv.org

---

*Built as a university mini-project. Feel free to extend and improve! 🚀*
=======
# Facial-Emotion-Recognition
Deep learning–based Facial Emotion Recognition system using CNN trained on the FER2013 Dataset, built with TensorFlow, Keras, and OpenCV. Supports image prediction and real-time webcam emotion detection.
>>>>>>> 924fc4b4afbde53d3fbce7a58effcf8c7676e519
