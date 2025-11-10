import os
import json
import zipfile
import subprocess
from pathlib import Path

# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Constants
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
EXTRACT_DIR = DATA_DIR / "plantvillage"
MODELS_DIR = BASE_DIR / "models"

# Kaggle dataset reference
KAGGLE_DATASET = "emmarex/plantdisease"  # PlantVillage dataset


def ensure_dirs():
    for d in [DATA_DIR, RAW_DIR, EXTRACT_DIR, MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def download_from_kaggle():
    """Download PlantVillage dataset via Kaggle API CLI."""
    # Requires Kaggle API credentials at ~/.kaggle/kaggle.json
    zip_path = RAW_DIR / "plantvillage.zip"
    if zip_path.exists():
        print(f"[skip] Zip already exists: {zip_path}")
        return zip_path

    print("[info] Downloading dataset from Kaggle...")
    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        KAGGLE_DATASET,
        "-p",
        str(RAW_DIR),
        "-o",
    ]
    subprocess.run(cmd, check=True)

    # The download may produce multiple files; find the largest .zip
    zips = sorted(RAW_DIR.glob("*.zip"), key=lambda p: p.stat().st_size, reverse=True)
    if not zips:
        raise RuntimeError("No zip file found after Kaggle download.")
    if zips[0] != zip_path:
        # Normalize to expected name
        zips[0].rename(zip_path)
    return zip_path


def extract_zip(zip_path: Path):
    if any(EXTRACT_DIR.iterdir()):
        print(f"[skip] Extract directory not empty: {EXTRACT_DIR}")
        return
    print(f"[info] Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(EXTRACT_DIR)


def build_datasets(img_size=(128, 128), batch_size=32, val_split=0.2, seed=42):
    """Create train/val datasets from extracted folder structure."""
    # Find the directory that contains class subfolders
    # Many PlantVillage zips contain a top-level directory with class subfolders inside
    root = EXTRACT_DIR
    # If there's a single subfolder that has the classes, descend into it
    subdirs = [p for p in root.iterdir() if p.is_dir()]
    if len(subdirs) == 1:
        root = subdirs[0]

    print(f"[info] Using dataset root: {root}")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        root,
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        root,
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
    )

    class_names = train_ds.class_names

    # Prefetch for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names


def build_model(num_classes: int, img_size=(128, 128)):
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
    ensure_dirs()
    zip_path = download_from_kaggle()
    extract_zip(zip_path)

    img_size = (128, 128)
    batch_size = 32
    epochs = 10

    train_ds, val_ds, class_names = build_datasets(img_size=img_size, batch_size=batch_size)

    print(f"[info] Classes: {len(class_names)} -> {class_names[:10]}{'...' if len(class_names) > 10 else ''}")

    model = build_model(num_classes=len(class_names), img_size=img_size)
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODELS_DIR / "crop_health_model.h5"),
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
        )
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)

    # Save labels mapping
    labels_path = MODELS_DIR / "labels.json"
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)
    print(f"[done] Saved model and labels to: {MODELS_DIR}")


if __name__ == "__main__":
    main()
