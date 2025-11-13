# Frontend (React + Vite) – Camera/Upload, Predict, and PDF Report

This frontend lets you capture an image from the camera or upload from disk, send it to the backend `/api/predict-health`, render a report, and download it as PDF.

## 1) Prerequisites
- Node.js 18+
- Backend running at `http://localhost:8000` (default) or your custom API URL

## 2) Environment variables
Copy `.env.sample` to `.env` if you want to override the API base URL.
```
# Example
VITE_API_BASE=http://localhost:8000
```
If `VITE_API_BASE` is not set, the app uses Vite proxy rules in `vite.config.js` to forward `/api` and `/health` to `http://localhost:8000` during development.

## 3) Install and run
```bash
cd frontend
npm install
npm run dev
```
Open: `http://localhost:5173`

## 4) Usage
- Click "Start Camera" and then "Capture" to take a photo, or select an image file.
- Click "Predict" to call the backend and view the detailed report.
- Click "Download Report (PDF)" to save the report.

## 5) Notes
- Camera access requires browser permission.
- The report shows `label_name` only if `backend/models/labels.json` exists.
- Ensure the backend model `backend/models/crop_health_model.h5` is present (or train via `backend/train_cnn.py`).
