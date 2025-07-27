from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse, FileResponse
import joblib
import os
import logging
from datetime import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# === Initialize FastAPI ===
app = FastAPI()

# === Paths ===
LOG_DIR = "logs"
STRUCTURED_LOG = os.path.join(LOG_DIR, "predictions.joblib")
PLOT_PATH = os.path.join(LOG_DIR, "survival_plot.png")

# === Clean logs and plot on startup ===
@app.on_event("startup")
def cleanup_logs_on_startup():
    os.makedirs(LOG_DIR, exist_ok=True)
    for file in [STRUCTURED_LOG, PLOT_PATH]:
        if os.path.exists(file):
            os.remove(file)

# === Load model and pipeline ===
model = joblib.load("titanic_model.pkl")
pipeline = joblib.load("model/pipeline.joblib")

# === File logger (optional) ===
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "predictions.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# === Pydantic schema ===
class Passenger(BaseModel):
    pclass: int
    sex: str
    age: float

# === Save structured logs ===
def log_structured_prediction(input_data, prediction):
    if os.path.exists(STRUCTURED_LOG):
        history = joblib.load(STRUCTURED_LOG)
    else:
        history = []
    history.append({
        "timestamp": datetime.now().isoformat(),
        "input": input_data,
        "prediction": int(prediction)
    })
    joblib.dump(history, STRUCTURED_LOG)

# === Predict endpoint ===
@app.post("/predict")
def predict(passenger: Passenger):
    sex_map = {"male": 0, "female": 1}
    sex_num = sex_map.get(passenger.sex.lower(), -1)

    if sex_num == -1:
        logging.warning(f"Invalid sex: {passenger.sex}")
        return {"error": "Invalid value for sex. Must be 'male' or 'female'."}

    input_data = [passenger.pclass, sex_num, passenger.age]
    prediction = pipeline.predict([input_data])[0]

    logging.info(f"Input: {input_data} → Prediction: {int(prediction)}")
    log_structured_prediction(input_data, prediction)

    return {"survived": int(prediction)}

# === Get logs as JSON ===
@app.get("/logs")
def get_prediction_logs():
    if not os.path.exists(STRUCTURED_LOG):
        return JSONResponse(content={"logs": []})
    try:
        logs = joblib.load(STRUCTURED_LOG)
        return {"logs": logs}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# === Plot endpoint ===
@app.get("/logs/plot")
def plot_survival_stats():
    if not os.path.exists(STRUCTURED_LOG):
        return {"error": "No logs found."}

    logs = joblib.load(STRUCTURED_LOG)
    df = pd.DataFrame(logs)

    if df.empty:
        return {"error": "No prediction data to plot."}

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='prediction', palette='Set2')
    plt.title("Survival Count")
    plt.xlabel("Prediction")
    plt.ylabel("Count")
    plt.tight_layout()

    plt.savefig(PLOT_PATH)
    plt.close()

    return FileResponse(PLOT_PATH, media_type="image/png", filename="survival_plot.png")

# === Optional: Reset everything manually ===
@app.delete("/reset")
def reset_logs():
    for file in [STRUCTURED_LOG, PLOT_PATH]:
        if os.path.exists(file):
            os.remove(file)
    return {"message": "Logs and plot reset successfully."}

