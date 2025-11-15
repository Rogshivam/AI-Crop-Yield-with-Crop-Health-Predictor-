import os
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# CIFAR-10 labels
CIFAR10_LABELS = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def build_model(num_classes: int, img_size=(32, 32)):
    inputs = keras.Input(shape=(*img_size, 3))
    x = layers.Rescaling(1.0/255)(inputs)
    x = layers.Conv2D(32, 3, activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load CIFAR-10 (downloads automatically)
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    y_train = y_train.squeeze().astype("int32")
    y_test = y_test.squeeze().astype("int32")

    # Simple train/val split
    val_count = 5000
    x_val, y_val = x_train[-val_count:], y_train[-val_count:]
    x_train, y_train = x_train[:-val_count], y_train[:-val_count]

    model = build_model(num_classes=10, img_size=(32, 32))

    # Keep it fast by default
    epochs = int(os.environ.get("EPOCHS", "3"))
    batch_size = int(os.environ.get("BATCH_SIZE", "64"))

    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
    )

    # Evaluate (optional)
    _ = model.evaluate(x_test, y_test, verbose=0)

    # Save model compatible with backend expectations
    model.save(MODELS_DIR / "crop_health_model.h5")
    with open(MODELS_DIR / "labels.json", "w", encoding="utf-8") as f:
        json.dump(CIFAR10_LABELS, f, ensure_ascii=False, indent=2)
    print("[done] Saved:", MODELS_DIR / "crop_health_model.h5", "and", MODELS_DIR / "labels.json")


if __name__ == "__main__":
    main()
