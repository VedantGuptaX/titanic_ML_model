itanic Survival Prediction App (FastAPI + Gradio)

This project is a full-stack ML web application that predicts whether a Titanic passenger would survive based on their input features (class, sex, and age).  
It includes a **FastAPI backend** for model inference and logging, and a **Gradio frontend** for user interaction.

---

Project Structure:

titanic_ml_app/
├── app/ # FastAPI backend (API logic, model, logging)
│ ├── main.py
│ ├── titanic_model.pkl
│ ├── model/
│ │ └── pipeline.joblib
│ └── logs/ # Logs saved here (predictions + plot)
│
├── gradio_ui/ # Gradio frontend (UI logic)
│ ├── gradio_ui.py
│ └── requirements.txt
│
├── docker-compose.yml # To spin up both services together
├── README.md

--- 


---

## 🚀 Features

- 🔮 **ML Prediction** using a scikit-learn pipeline
- 📈 **Logs predictions** in a `.joblib` file and text logs
- 🖼️ **Survival Plot** auto-generated from logs using Seaborn/Matplotlib
- 🧪 **Interactive UI** built with Gradio
- 🐳 **Dockerized**, ready for production or local testing

---

## 🧪 How It Works

- You input a **Passenger Class (1, 2, 3)**, **Sex**, and **Age** in the Gradio UI.
- The UI sends a POST request to the FastAPI `/predict` endpoint.
- The backend:
  - Converts inputs
  - Makes prediction using the pre-trained model
  - Logs input and prediction to `logs/predictions.joblib`
  - Returns survival status
- A survival plot is available at `/logs/plot` or automatically shown in UI.

---

Ports Used:

| Service   | Port | Description                        |
|-----------|------|------------------------------------|
| FastAPI   | 8000 | Backend API for prediction/logging |
| Gradio UI | 7860 | Frontend interface for interaction |

---

1. Start the Backend

cd app
uvicorn main:app --reload --port 8000 

2. Start the Frontend

cd gradio_ui
python gradio_ui.py
For local testing (Without Docker and K8s) 

---

API Endpoints (FastAPI)

POST - /predict (Submit passenger data for prediction)
GET	- /logs	(Get all structured logs (JSON))
GET	- /logs/plot	(Get survival plot as PNG image)

---

🧹 Cleanup Between Runs
If logs or plot files interfere with testing, delete:
rm app/logs/predictions.joblib
rm app/logs/survival_plot.png

---



VedantGuptaX
Developed by Vedant
