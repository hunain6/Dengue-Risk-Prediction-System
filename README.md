# Dengue-Risk-Prediction-System
A Machine Learning–powered system for predicting dengue risk levels using weather and dengue data (2016–2024), with predictions for the year 2025 and for the yaer 2026.
Includes:

ML Model (Python + Scikit-Learn, Time serie Forecasting)
FastAPI Backend
Flutter Mobile App Frontend
REST API for real-time dengue risk prediction


📌 1. Project Overview

Dengue is a growing health concern influenced by weather conditions such as rainfall, temperature, humidity, and seasonal patterns.
This project uses historical data (2016–2024) to:

✔ Train a dengue risk clustering model
✔ Predict and classify dengue risk for 2025 and 2026 (all months & regions)
✔ Expose a prediction API using FastAPI
✔ Provide a Flutter mobile app for easy user interaction

This system helps researchers, health departments, and the public better understand and forecast dengue risk.


📊 2. Dataset Details

Data Range: 2016 — 2024
Features Used:

Dengue Cases

Temperature

Rainfall

Humidity

Deaths

Other weather factors

After training the model, the system predicts:
✔ Risk score
✔ Risk category (Low, Medium, High)
✔ 2025 predictions for all months & regions


🤖 3. Machine Learning Model
Algorithms Used:

K-Means Clustering

PCA (Principal Component Analysis) for visualization & dimensionality reduction

Time series Forecasting for 2026 dengue data

StandardScaler for feature scaling

Model Outputs:

Cluster category (0, 1, 2 …)

Risk score based on distance from centroids

Future risk predictions (2025)



⚙️ 4. Backend (FastAPI)

The backend provides APIs for:

✔ Predicting dengue risk
✔ Getting saved predictions
✔ Health checks
📱 5. Frontend (Flutter App)

Features:
✔ Form for input values
✔ Sends data to FastAPI backend
✔ Displays predicted category & risk
✔ Shows 2025, 2026 monthly risk charts
✔ Modern and responsive UI



🛠️ 7. How to Configure Everything
Step 1 — Train Model (Optional)

If you want to retrain:

run model_training.ipynb

Step 2 — Start FastAPI Backend
uvicorn main:app --reload

Step 3 — Run Flutter App
flutter run

Step 4 — Connect App & Backend

Ensure both are on same network or use:

WiFi IP address

OR Deploy backend online (Render / AWS / Railway)


Prediction Page:


![WhatsApp Image 2025-12-12 at 9 05 33 PM](https://github.com/user-attachments/assets/3f4fe950-6c2d-4ba6-9d59-01b8237de6ad)




🎯 10. Future Work

District-based predictions

Adding mosquito density data

Real-time weather API integration

Deploying mobile app to Play Store
