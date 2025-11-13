# Backend (Flask) – Setup and Usage

This backend serves two endpoints:
- `POST /api/predict-health` – Classify leaf image via CNN model
- `POST /api/predict-yield` – Predict yield via scikit-learn model
- `GET /health` – Service/model status

## 1) Prerequisites
- Python 3.10+ recommended
- Kaggle account (for training CNN)

## 2) Environment variables
Copy `.env.sample` to `.env` and adjust values if needed.
```
PORT=8000
KAGGLE_USERNAME=
KAGGLE_KEY=
```
You may also put `kaggle.json` in your user profile (Windows: `%USERPROFILE%\.kaggle\kaggle.json`, WSL/Linux: `~/.kaggle/kaggle.json`).

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

Optional: On CPU-only machines, installing `tensorflow-cpu==2.12.0` can be faster than full `tensorflow`.

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

## 5) Datasets: Kaggle vs KaggleHub

- Yield dataset can be fetched without API keys using KaggleHub:
```bash
python backend/download_yield_dataset.py
```
It will copy files into `backend/data/yield/`.

- CNN PlantVillage dataset download and training uses the Kaggle API (requires API key or `kaggle.json`).

## 6) Train the CNN from Kaggle (optional)
This downloads PlantVillage, trains, and saves the model/labels into `backend/models/`.
```bash
python backend/train_cnn.py
```
Outputs:
- `backend/models/crop_health_model.h5`
- `backend/models/labels.json`

## 7) Run the API
```bash
python backend/app.py
```
Server: `http://localhost:8000`

Health:
```bash
curl http://localhost:8000/health
```

## 8) Models
Place exported models in `backend/models/`:
- `yield_model.pkl`
- `crop_health_model.h5`
- `labels.json` (optional, for human-readable names)

## 9) Secrets and safety
- Do not commit API keys into the repository. Prefer using user-profile `kaggle.json` or shell environment variables.
- If you accidentally exposed a Kaggle key, rotate it on Kaggle immediately.
