import os
from pathlib import Path
import kagglehub

# Download latest version of the crop yield dataset using kagglehub
# Dataset: https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset
# No Kaggle API key required for kagglehub downloads.

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "yield"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def main():
    path = kagglehub.dataset_download("patelris/crop-yield-prediction-dataset")
    print("Path to dataset files:", path)
    # Optionally copy into our project data folder for convenience
    # This avoids dealing with KaggleHub cache locations elsewhere.
    try:
        # Only copy if not already copied
        src = Path(path)
        dst = DATA_DIR
        if not any(dst.iterdir()):
            import shutil
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print("Copied dataset into:", dst)
        else:
            print("Dataset directory already populated:", dst)
    except Exception as e:
        print("Copy skipped due to:", e)


if __name__ == "__main__":
    main()
