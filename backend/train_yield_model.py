import os
import csv
import re
import json
import shutil
from typing import List, Dict, Optional

import numpy as np

try:
    import joblib  # type: ignore
except Exception:
    joblib = None

# Optional: use kagglehub if available to fetch the dataset automatically
try:
    import kagglehub  # type: ignore
except Exception:
    kagglehub = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "yield")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "yield_model.pkl")

# Feature aliases mapping (case-insensitive, punctuation-insensitive)
ALIASES: Dict[str, List[str]] = {
    "rainfall": [
        "rainfall",
        "average rainfall (mm/yr)",
        "avg_rainfall",
        "average_rainfall",
        "avg rainfall",
    ],
    "temperature": [
        "temperature",
        "avg_temp",
        "average temperature (celsius)",
        "average_temperature",
        "avg temperature",
    ],
    "soil_ph": [
        "soil_ph",
        "soil ph",
        "ph",
    ],
    "fertilizer": [
        "fertilizer",
        "fertilizer(kg/ha)",
        "fertilizer use (kg/ha)",
        "fertilizer_use",
    ],
    "area": [
        "area",
        "area (hectares)",
        "hectares",
    ],
    "yield": [
        "yield",
        "yield (kg/ha)",
        "crop yield (kg/ha)",
        "crop_yield",
        "production",
    ],
}


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.strip().lower()).strip()


def find_csv_in_dir(directory: str) -> Optional[str]:
    if not os.path.isdir(directory):
        return None
    candidates = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(".csv"):
                candidates.append(os.path.join(root, f))
    if not candidates:
        return None
    # Prefer files with 'yield' in the name
    for p in candidates:
        if "yield" in os.path.basename(p).lower():
            return p
    return candidates[0]


def maybe_fetch_with_kagglehub() -> Optional[str]:
    if kagglehub is None:
        return None
    try:
        path = kagglehub.dataset_download("patelris/crop-yield-prediction-dataset")
        # Copy csvs into DATA_DIR if not present
        os.makedirs(DATA_DIR, exist_ok=True)
        copied = None
        for root, _, files in os.walk(path):
            for f in files:
                if f.lower().endswith(".csv"):
                    src = os.path.join(root, f)
                    dst = os.path.join(DATA_DIR, f)
                    if not os.path.exists(dst):
                        shutil.copy2(src, dst)
                    if copied is None:
                        copied = dst
        return copied
    except Exception:
        return None


def load_table(csv_path: str) -> List[Dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [dict(r) for r in reader]
    return rows


def map_columns(headers: List[str]) -> Dict[str, str]:
    # returns mapping from canonical to actual header name
    h_norm = {norm(h): h for h in headers}
    mapping: Dict[str, str] = {}
    for canon, alias_list in ALIASES.items():
        for alias in alias_list:
            key = norm(alias)
            if key in h_norm:
                mapping[canon] = h_norm[key]
                break
    return mapping


def to_float(x: str) -> Optional[float]:
    if x is None:
        return None
    try:
        # remove commas and spaces
        return float(str(x).replace(",", "").strip())
    except Exception:
        return None


class LinearYieldModel:
    def __init__(self, coef: np.ndarray, intercept: float, feature_order: List[str]):
        self.coef = np.asarray(coef, dtype=float).reshape(-1)
        self.intercept = float(intercept)
        self.feature_order = list(feature_order)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return X.dot(self.coef) + self.intercept


def train_and_save(csv_path: str) -> None:
    rows = load_table(csv_path)
    if not rows:
        raise RuntimeError("CSV appears empty: " + csv_path)

    headers = list(rows[0].keys())
    mapping = map_columns(headers)

    required = ["rainfall", "temperature", "soil_ph", "fertilizer", "area", "yield"]
    missing = [k for k in required if k not in mapping]
    if missing:
        raise RuntimeError(f"Could not find required columns in CSV: {missing}. Found mapping: {json.dumps(mapping)}")

    feats = ["rainfall", "temperature", "soil_ph", "fertilizer", "area"]
    X_list: List[List[float]] = []
    y_list: List[float] = []

    for r in rows:
        vals: List[Optional[float]] = [to_float(r.get(mapping[k])) for k in feats]
        yval = to_float(r.get(mapping["yield"]))
        if any(v is None for v in vals) or yval is None:
            continue
        X_list.append([v for v in vals if v is not None])
        y_list.append(yval)

    if len(X_list) < 10:
        raise RuntimeError("Not enough valid rows after cleaning to train a model.")

    X = np.asarray(X_list, dtype=float)
    y = np.asarray(y_list, dtype=float)

    # Fit linear regression using least squares: y = Xb + c
    # Augment X with bias term to get intercept
    X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
    beta, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
    coef, intercept = beta[:-1], beta[-1]

    os.makedirs(MODELS_DIR, exist_ok=True)
    model = LinearYieldModel(coef=coef, intercept=intercept, feature_order=feats)
    if joblib is None:
        raise RuntimeError("joblib is required to save the model. Install joblib in your environment.")
    joblib.dump(model, MODEL_PATH)
    print(f"Saved yield model to {MODEL_PATH}")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    csv_path = find_csv_in_dir(DATA_DIR)
    if csv_path is None:
        print("No CSV found in data/yield. Attempting to fetch with kagglehub...")
        csv_path = maybe_fetch_with_kagglehub()
        if csv_path is None:
            raise SystemExit("CSV not found. Place dataset CSV under backend/data/yield/ or install kagglehub and re-run.")
    print("Using CSV:", csv_path)
    train_and_save(csv_path)


if __name__ == "__main__":
    main()
