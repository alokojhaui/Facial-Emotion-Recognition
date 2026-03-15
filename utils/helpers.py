"""
============================================================
  utils/helpers.py  —  Shared utilities
============================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from sklearn.metrics import classification_report, confusion_matrix


EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


# ─────────────────────────────────────────────
# DATASET STATS
# ─────────────────────────────────────────────
def plot_class_distribution(y_train, y_val=None, y_test=None):
    """
    Bar chart showing sample counts per emotion class in each split.
    Highlights the class imbalance in FER2013 (Disgust is very rare).
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(EMOTION_LABELS))
    w = 0.25

    counts_train = [np.sum(y_train == i) for i in range(len(EMOTION_LABELS))]
    ax.bar(x - w, counts_train, width=w, label='Train', color='royalblue', alpha=0.85)

    if y_val is not None:
        counts_val = [np.sum(y_val == i) for i in range(len(EMOTION_LABELS))]
        ax.bar(x, counts_val, width=w, label='Validation', color='orange', alpha=0.85)

    if y_test is not None:
        counts_test = [np.sum(y_test == i) for i in range(len(EMOTION_LABELS))]
        ax.bar(x + w, counts_test, width=w, label='Test', color='green', alpha=0.85)

    ax.set_xticks(x); ax.set_xticklabels(EMOTION_LABELS, rotation=20)
    ax.set_ylabel('Sample Count'); ax.set_title('Class Distribution per Split')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/class_distribution.png', dpi=150)
    plt.show()
    print("[INFO] Class distribution saved → results/class_distribution.png")


# ─────────────────────────────────────────────
# VISUALISE SAMPLE IMAGES
# ─────────────────────────────────────────────
def show_sample_images(X, y, n_per_class=4):
    """Show n_per_class sample images for each emotion label."""
    fig, axes = plt.subplots(len(EMOTION_LABELS), n_per_class,
                             figsize=(n_per_class * 2, len(EMOTION_LABELS) * 2))
    for cls_idx, label in enumerate(EMOTION_LABELS):
        indices = np.where(y == cls_idx)[0]
        chosen  = np.random.choice(indices, n_per_class, replace=False)
        for j, idx in enumerate(chosen):
            axes[cls_idx][j].imshow(X[idx].squeeze(), cmap='gray')
            axes[cls_idx][j].axis('off')
            if j == 0:
                axes[cls_idx][j].set_ylabel(label, fontsize=10, rotation=0,
                                             labelpad=45, va='center')
    plt.suptitle('Sample Images per Emotion Class', fontsize=13)
    plt.tight_layout()
    plt.savefig('results/sample_images_per_class.png', dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# GRAD-CAM (simple version for last conv layer)
# ─────────────────────────────────────────────
def grad_cam(model, img_array, layer_name=None):
    """
    Compute Grad-CAM heatmap for a single preprocessed image (1,48,48,1).
    If layer_name is None, uses the last Conv2D layer automatically.
    Returns the heatmap (48×48 float32).
    """
    import tensorflow as tf

    if layer_name is None:
        # Find last Conv2D layer
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer_name = layer.name
                break

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(original_img_gray, heatmap):
    """Overlay Grad-CAM heatmap on the original grayscale face image."""
    img_rgb  = cv2.cvtColor(
        (original_img_gray * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR
    )
    heatmap_resized = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    superimposed    = cv2.addWeighted(img_rgb, 0.6, heatmap_colored, 0.4, 0)
    return cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
