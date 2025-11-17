# Backend (Flask) – Setup and Usage

Service endpoints:
- `POST /api/predict` – Classify an image using the CNN model (also available as `/api/predict-health` for backward compatibility)
- `GET /health` – Returns API and model/labels status flags
- `POST /admin/reload-models` – Reload models/labels from disk without restarting the server

## 1) Prerequisites
- Python 3.10+ recommended
- CPU is fine; GPU not required
- Optional: Kaggle account only if you want to train on PlantVillage

## 2) Environment variables
Copy `.env.sample` to `.env` and adjust values if needed.
```
PORT=8000
```
If you train with Kaggle later, place `kaggle.json` at:
- Windows: `%USERPROFILE%\.kaggle\kaggle.json`
- WSL/Linux: `~/.kaggle/kaggle.json` (run `chmod 600 ~/.kaggle/kaggle.json`)

## 3) Create a virtual environment
Pick ONE path depending on your shell/OS.

- Windows PowerShell (native Windows):
```powershell
py --list            # optional: see installed Python versions
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
```

- WSL / Linux / macOS:
```bash
sudo apt update && sudo apt install -y python3-venv python3-full  # if needed
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

If you see an "externally managed" or PEP 668 error, you are using system Python without an active venv. Activate the venv and use its pip, e.g. `python -m pip install ...`.

## 4) Install dependencies
From the project root or from `backend/`:
```bash
python -m pip install -r backend/requirements.txt
```

WSL/Linux tips:
- Avoid interrupting pip on first run; building packages (e.g., TensorFlow) can take a while.
- If downloads hang, set a longer timeout and no-cache:
```bash
python -m pip install --default-timeout=120 --no-cache-dir -r backend/requirements.txt
```
- Still hitting system-Python issues? As a last resort only:
```bash
python -m pip install --break-system-packages -r backend/requirements.txt
```

Note: This project uses the CPU build of TensorFlow by default.

### WSL/Linux troubleshooting (venv/pip hangs)
- If venv creation was interrupted (KeyboardInterrupt), delete and recreate it:
```bash
deactivate 2>/dev/null || true
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
python -m ensurepip --upgrade
python -m pip install --upgrade pip
python -m pip install --default-timeout=120 --no-cache-dir -r backend/requirements.txt
```
- If you are working under a Windows-mounted path like `/mnt/c/...`, installs can be slow. Consider cloning the repo into your Linux home directory (e.g., `~/projects/...`) for better performance.

## 5) Models and datasets – two easy paths

Pick ONE path to obtain a working CNN model + labels.

### Path A – Quick local model (NO Kaggle)
Train a small CNN on CIFAR‑10. Data is downloaded automatically by Keras.
```bash
cd backend
# Use your venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --no-cache-dir "tensorflow-cpu>=2.20,<2.21" numpy pillow

# Train fast (set EPOCHS=1..3)
EPOCHS=2 python train_classifier.py
```
Outputs (created under `backend/models/`):
- `crop_health_model.h5`
- `labels.json` (CIFAR‑10 class names)

### Path B – PlantVillage (Kaggle)
Use the Kaggle API to download the PlantVillage dataset and train a CNN.
```bash
# One-time Kaggle setup
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

cd backend
source .venv/bin/activate
python -m pip install --no-cache-dir kaggle tqdm "tensorflow-cpu>=2.20,<2.21" numpy pillow
python train_cnn.py   # downloads data, trains, saves best model
```
Outputs (created under `backend/models/`):
- `crop_health_model.h5`
- `labels.json` (PlantVillage class names)

Dataset folders used by the trainer (auto-managed):
- `backend/data/raw` – downloaded zip(s)
- `backend/data/plantvillage` – extracted images

## 6) Optional – Yield model
If you also need a numeric yield predictor, a simple regression trainer is available:
```bash
cd backend
python -m pip install --no-cache-dir joblib numpy kagglehub
python train_yield_model.py
```
Output:
- `backend/models/yield_model.pkl`

## 7) Run the API
```bash
python backend/app.py
```
Server: `http://localhost:8000`

Health:
```bash
curl http://localhost:8000/health
```
You should see flags such as:
```
{
  "status": "ok",
  "health_model_loaded": true,
  "health_labels_loaded": true,
  "health_model_file_exists": true,
  "labels_file_exists": true
}
```

Reload models without restart:
```bash
curl -X POST http://localhost:8000/admin/reload-models
```

## 8) Model file locations
Place files under `backend/models/` (when you’re cd’ed into `backend/`, that is `models/`):
- `crop_health_model.h5` (required for image prediction)
- `labels.json` (recommended; used to map label indices to names)
- `yield_model.pkl` (optional; only if you use `/api/predict-yield`)

## 9) Secrets and safety
- Do not commit API keys into the repository. Prefer using user-profile `kaggle.json` or shell environment variables.
- If you accidentally exposed a Kaggle key, rotate it on Kaggle immediately.
