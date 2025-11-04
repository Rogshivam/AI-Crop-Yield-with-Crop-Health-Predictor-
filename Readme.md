# 🌱 AI Crop Yield Predictor + Crop Health & Sustainability Dashboard

## 🧭 Overview

An AI-powered sustainability web application that:
- Predicts **crop yield** using soil, weather, and fertilizer data.
- Detects **crop health** from leaf images using a **Convolutional Neural Network (CNN)**.
- Displays all insights on a **React-based Sustainability Dashboard** with visual analytics, reports, and AI recommendations.

This project integrates **Machine Learning**, **Deep Learning (CNN)**, and a **full-stack architecture** (React + Flask/Express + MongoDB) to promote **data-driven sustainable farming**.

---

## 🌍 Sustainability Impact

🌾 Helps farmers optimize crop yield.  
💧 Reduces overuse of fertilizer and water.  
🍃 Enables early disease detection for healthier crops.  
🔋 Supports sustainable agriculture through AI-driven insights.  

---

## 🧱 System Architecture

```plaintext
                   ┌──────────────────────────────┐
                   │        React Frontend         │
                   │  • Upload Leaf Image          │
                   │  • Enter Weather/Soil Data    │
                   │  • View Dashboard & Charts    │
                   └──────────────┬────────────────┘
                                  │
                                  ▼
                   ┌──────────────────────────────┐
                   │     Express / Flask Backend   │
                   │  • /api/predict-health        │
                   │  • /api/predict-yield         │
                   │  • Calls Python ML Models     │
                   └──────────────┬────────────────┘
                                  │
                                  ▼
                   ┌──────────────────────────────┐
                   │     AI Models (Python)        │
                   │  • CNN (TensorFlow/Keras)     │
                   │  • Regression (scikit-learn)  │
                   └──────────────┬────────────────┘
                                  │
                                  ▼
                   ┌──────────;────────────────────┐
                   │     MongoDB / PostgreSQL      │
                   │  • User & Prediction Data     │
                   └──────────────────────────────┘
```
## 🧠 AI Models
### 🔹 Crop Health Detection (CNN)

Dataset:
PlantVillage Dataset (Kaggle) : https://www.kaggle.com/datasets/emmarex/plantdisease 
Crop Yield Prediction Dataset (Kaggle) : https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset

Goal: Classify crop leaf images as Healthy or Diseased (e.g., Tomato Bacterial Spot, Potato Late Blight).

Model Architecture (TensorFlow/Keras):
```bash
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])
```

output :-
```
{
  "crop": "Tomato",
  "status": "Diseased",
  "disease": "Bacterial Spot",
  "confidence": 0.94
}
```
### 🔹 Crop Yield Prediction (Regression)

Dataset:
Crop Yield Prediction Dataset (Kaggle)

Goal: Predict yield (tons/hectare) based on soil, rainfall, and temperature data.

Model Example (Random Forest):
```bash
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, 'models/yield_model.pkl')
```
Input Features Example:

Feature	Description
rainfall	Average rainfall (mm)
temperature	Average temperature (°C)
soil_ph	Soil pH value
fertilizer	Amount used (kg/ha)
area	Cultivation area (hectares)

## 🌐 Backend API (Flask Example)
```bash
@app.route('/api/predict-yield', methods=['POST'])
def predict_yield():
    data = request.json
    X = [[data['rainfall'], data['temperature'], data['soil_ph'], data['fertilizer'], data['area']]]
    prediction = yield_model.predict(X)[0]
    return jsonify({"predicted_yield": round(prediction, 2)})

@app.route('/api/predict-health', methods=['POST'])
def predict_health():
    file = request.files['image']
    img = image.load_img(file, target_size=(128,128))
    img_array = np.expand_dims(image.img_to_array(img)/255.0, axis=0)
    preds = health_model.predict(img_array)
    label = np.argmax(preds)
    confidence = np.max(preds)
    return jsonify({"label": int(label), "confidence": float(confidence)})
```
## 💾 Database Structure (MongoDB)
Collections:

* users → { name, email, password_hash }

* predictions → { user_id, type, input_data, output_data, timestamp }

Example Document:
```
{
  "type": "yield",
  "input_data": { "crop": "Wheat", "rainfall": 450, "temperature": 27 },
  "output_data": { "yield": 3.42 },
  "timestamp": "2025-11-04T12:45:00Z"
}
```

## 🖥️ Frontend (React + Tailwind CSS)
Pages

1. Home Page – Project overview & sustainability mission

2. Yield Predictor – Input soil, fertilizer, and weather data → get yield prediction

3. Crop Health Page – Upload leaf image → CNN detects health & disease

4. Dashboard – Charts for:

* * Crop health distribution

* * Yield trends

Sustainability scores

5. Reports Page – Generate AI-based recommendations and export to PDF
Libraries Used:

* React.js

* Tailwind CSS

* Chart.js / Recharts

* Axios

* React Router

## 📊 Dashboard Visualization Ideas
| Widget                    | Description                                   |
| ------------------------- | --------------------------------------------- |
| 📈 Yield Prediction Chart | Shows predicted yield over time               |
| 🍃 Health Analysis        | Pie chart of healthy vs diseased crops        |
| 🌤️ Weather Data          | Live weather input integration                |
| 🌱 Sustainability Score   | Combines yield + health + environment metrics |


# ⚙️ Tech Stack
| Layer             | Tool                                  | Purpose                              |
| ----------------- | ------------------------------------- | ------------------------------------ |
| **Frontend**      | React.js + Tailwind CSS               | Dashboard & visualization            |
| **Backend**       | Flask / Express.js                    | API & ML model serving               |
| **AI Models**     | TensorFlow, Keras, scikit-learn       | CNN + Regression                     |
| **Database**      | MongoDB Atlas                         | Store user data & predictions        |
| **Visualization** | Chart.js / Recharts                   | Graphs and charts                    |
| **Data Source**   | Kaggle Datasets                       | Crop yield & health datasets         |
| **Hosting**       | Render / Vercel / Hugging Face Spaces | Deployment                           |
| **APIs**          | OpenWeatherMap / GPT API              | Weather & sustainability suggestions |

## 📦 ai-crop-
```bash 
├── backend/
│   ├── app.py
│   ├── models/
│   │   ├── yield_model.pkl
│   │   └── crop_health_model.h5
│   ├── routes/
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   └── App.jsx
│   └── package.json
└── README.md

📦 ai-crop-sustainability
├── backend/
│   ├── app.py
│   ├── models/
│   │   ├── yield_model.pkl
│   │   └── crop_health_model.h5
│   ├── routes/
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   └── App.jsx
│   └── package.json
└── README.md
```


## 🧩 Future Enhancements

✅ Integrate GPT API for sustainability tips
✅ Add weather-based real-time yield predictions
✅ Add map-based crop visualization (Leaflet.js)
✅ Implement TensorFlow.js for in-browser CNN inference
✅ Export sustainability reports as PDF

## 📘 References

PlantVillage Dataset – Kaggle

Crop Yield Prediction Dataset – Kaggle

OpenWeatherMap API

TensorFlow / scikit-learn Documentation

## 🧑‍💻 Author

Rogshivam(Shivam Kumar)
*🌾 Passionate about AI, sustainability, and smart agriculture solutions.
*📧 Feel free to connect or contribute!