"""
============================================================
  Facial Emotion Recognition using CNN  |  train.py
  Dataset: FER2013  |  Framework: TensorFlow / Keras
============================================================
"""

# ─────────────────────────────────────────────
# 1. IMPORT LIBRARIES
# ─────────────────────────────────────────────
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix

# Reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────
# 2. CONSTANTS & CONFIGURATION
# ─────────────────────────────────────────────
IMG_SIZE    = 48          # FER2013 images are 48×48 pixels
BATCH_SIZE  = 64
EPOCHS      = 60
NUM_CLASSES = 7
LR          = 1e-3

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Paths
DATA_CSV   = 'data/fer2013.csv'
MODEL_PATH = 'models/emotion_cnn_best.keras'
HISTORY_PATH = 'results/training_history.npy'

# ─────────────────────────────────────────────
# 3. LOAD & PREPROCESS FER2013 DATASET
# ─────────────────────────────────────────────
def load_fer2013(csv_path):
    """
    FER2013 CSV has three columns:
        emotion  (0-6 integer label)
        pixels   (space-separated 48×48 = 2304 values, 0-255)
        Usage    ('Training', 'PublicTest', 'PrivateTest')
    """
    print(f"[INFO] Loading dataset from '{csv_path}' ...")
    df = pd.read_csv(csv_path)

    def row_to_image(pixel_str):
        arr = np.array(pixel_str.split(), dtype=np.float32)
        return arr.reshape(IMG_SIZE, IMG_SIZE, 1) / 255.0   # normalise to [0,1]

    X, y = [], []
    for _, row in df.iterrows():
        X.append(row_to_image(row['pixels']))
        y.append(int(row['emotion']))

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    # Split by 'Usage' column
    train_mask = df['Usage'] == 'Training'
    val_mask   = df['Usage'] == 'PublicTest'
    test_mask  = df['Usage'] == 'PrivateTest'

    X_train, y_train = X[train_mask], y[train_mask]
    X_val,   y_val   = X[val_mask],   y[val_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    print(f"  Train : {X_train.shape[0]} samples")
    print(f"  Val   : {X_val.shape[0]}   samples")
    print(f"  Test  : {X_test.shape[0]}  samples")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# ─────────────────────────────────────────────
# 4. DATA AUGMENTATION
# ─────────────────────────────────────────────
def build_data_generators(X_train, y_train, X_val, y_val):
    """
    Augment training data to improve generalisation and reduce overfitting.
    Validation data is NOT augmented — only rescaled (already normalised here).
    """
    train_datagen = ImageDataGenerator(
        rotation_range=15,        # random rotations ±15°
        width_shift_range=0.1,    # horizontal shift up to 10%
        height_shift_range=0.1,   # vertical shift up to 10%
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,     # faces are symmetric
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator()   # no augmentation for validation

    train_gen = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
    val_gen   = val_datagen.flow(X_val,   y_val,   batch_size=BATCH_SIZE, shuffle=False)

    return train_gen, val_gen


# ─────────────────────────────────────────────
# 5. CNN ARCHITECTURE
# ─────────────────────────────────────────────
def build_model(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=NUM_CLASSES):
    """
    Deep CNN with:
      • 4 convolutional blocks (Conv → BN → Conv → BN → MaxPool → Dropout)
      • Global Average Pooling (lighter than Flatten for regularisation)
      • 2 Dense layers with Dropout
      • Softmax output for 7 emotion classes

    Architecture inspired by VGG-style blocks but adapted for 48×48 grayscale input.
    """
    model = models.Sequential([
        # ── Input ──────────────────────────────────────────────────
        layers.Input(shape=input_shape),

        # ── Block 1 : 64 filters ───────────────────────────────────
        layers.Conv2D(64, (3,3), padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        # ── Block 2 : 128 filters ──────────────────────────────────
        layers.Conv2D(128, (3,3), padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3,3), padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        # ── Block 3 : 256 filters ──────────────────────────────────
        layers.Conv2D(256, (3,3), padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3,3), padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        # ── Block 4 : 512 filters ──────────────────────────────────
        layers.Conv2D(512, (3,3), padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Conv2D(512, (3,3), padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        # ── Classifier Head ────────────────────────────────────────
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=LR),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ─────────────────────────────────────────────
# 6. CALLBACKS
# ─────────────────────────────────────────────
def get_callbacks():
    os.makedirs('models',  exist_ok=True)
    os.makedirs('results', exist_ok=True)

    early_stop = EarlyStopping(
        monitor='val_accuracy', patience=12,
        restore_best_weights=True, verbose=1
    )
    checkpoint = ModelCheckpoint(
        MODEL_PATH, monitor='val_accuracy',
        save_best_only=True, verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=5, min_lr=1e-6, verbose=1
    )
    tensorboard = TensorBoard(log_dir='results/logs', histogram_freq=1)

    return [early_stop, checkpoint, reduce_lr, tensorboard]


# ─────────────────────────────────────────────
# 7. VISUALISE TRAINING HISTORY
# ─────────────────────────────────────────────
def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history['accuracy'],     label='Train Accuracy', color='royalblue')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy',   color='tomato')
    axes[0].set_title('Model Accuracy', fontsize=14)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history['loss'],     label='Train Loss', color='royalblue')
    axes[1].plot(history.history['val_loss'], label='Val Loss',   color='tomato')
    axes[1].set_title('Model Loss', fontsize=14)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/training_history.png', dpi=150)
    plt.show()
    print("[INFO] Training history saved → results/training_history.png")


# ─────────────────────────────────────────────
# 8. EVALUATE & CONFUSION MATRIX
# ─────────────────────────────────────────────
def evaluate_model(model, X_test, y_test):
    print("\n[INFO] Evaluating on test set ...")
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Test Loss     : {loss:.4f}")
    print(f"  Test Accuracy : {acc*100:.2f}%")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    # Classification report
    print("\n[INFO] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=EMOTION_LABELS))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS
    )
    plt.title('Confusion Matrix — Test Set', fontsize=14)
    plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png', dpi=150)
    plt.show()
    print("[INFO] Confusion matrix saved → results/confusion_matrix.png")
    return acc


# ─────────────────────────────────────────────
# 9. SAMPLE PREDICTIONS VISUALISATION
# ─────────────────────────────────────────────
def visualise_predictions(model, X_test, y_test, n=16):
    idx = np.random.choice(len(X_test), n, replace=False)
    X_sample = X_test[idx]
    y_true   = y_test[idx]
    y_pred   = np.argmax(model.predict(X_sample, verbose=0), axis=1)

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(X_sample[i].squeeze(), cmap='gray')
        color = 'green' if y_pred[i] == y_true[i] else 'red'
        ax.set_title(
            f"True: {EMOTION_LABELS[y_true[i]]}\nPred: {EMOTION_LABELS[y_pred[i]]}",
            color=color, fontsize=9
        )
        ax.axis('off')
    plt.suptitle('Sample Predictions (Green=Correct, Red=Wrong)', fontsize=13)
    plt.tight_layout()
    plt.savefig('results/sample_predictions.png', dpi=150)
    plt.show()
    print("[INFO] Sample predictions saved → results/sample_predictions.png")


# ─────────────────────────────────────────────
# 10. MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Facial Emotion Recognition — CNN Training")
    print("=" * 60)

    # ── Load data ───────────────────────────────────────
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_fer2013(DATA_CSV)

    # ── Augmented generators ─────────────────────────────
    train_gen, val_gen = build_data_generators(X_train, y_train, X_val, y_val)

    # ── Build model ──────────────────────────────────────
    model = build_model()
    model.summary()

    # ── Train ────────────────────────────────────────────
    steps_per_epoch  = len(X_train) // BATCH_SIZE
    validation_steps = len(X_val)   // BATCH_SIZE

    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=get_callbacks(),
        verbose=1
    )

    # Save history
    np.save(HISTORY_PATH, history.history)
    print(f"[INFO] Training history saved → {HISTORY_PATH}")

    # ── Plots ────────────────────────────────────────────
    plot_history(history)

    # ── Evaluate ─────────────────────────────────────────
    # Load best weights (already restored by EarlyStopping)
    evaluate_model(model, X_test, y_test)

    # ── Sample predictions ───────────────────────────────
    visualise_predictions(model, X_test, y_test)

    print("\n[DONE] Training complete. Best model saved →", MODEL_PATH)


if __name__ == '__main__':
    main()
