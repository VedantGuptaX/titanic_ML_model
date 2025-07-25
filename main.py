from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import logging
from datetime import datetime

# Load model and pipeline
model = joblib.load("titanic_model.pkl")
pipeline = joblib.load("model/pipeline.joblib")

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    filename="logs/predictions.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize FastAPI app
app = FastAPI()

# Pydantic schema
class Passenger(BaseModel):
    pclass: int
    sex: str  # 'male' or 'female'
    age: float

# Structured log function (joblib)
def log_structured_prediction(input_data, prediction):
    log_file = "logs/predictions.joblib"
    if os.path.exists(log_file):
        history = joblib.load(log_file)
    else:
        history = []

    history.append({
        "timestamp": datetime.now().isoformat(),
        "input": input_data,
        "prediction": int(prediction)
    })

    joblib.dump(history, log_file)

# Prediction route
@app.post("/predict")
def predict(passenger: Passenger):
    # Convert 'sex' to numeric
    sex_map = {"male": 0, "female": 1}
    sex_num = sex_map.get(passenger.sex.lower(), -1)

    if sex_num == -1:
        logging.warning(f"Invalid input: sex={passenger.sex}")
        return {"error": "Invalid value for sex. Must be 'male' or 'female'."}

    # Prepare input
    input_data = [passenger.pclass, sex_num, passenger.age]
    prediction = pipeline.predict([input_data])[0]

    # Log to file
    logging.info(f"Input: {input_data} → Prediction: {int(prediction)}")

    # Log structured prediction
    log_structured_prediction(input_data, prediction)
    
    
from fastapi.responses import JSONResponse

@app.get("/logs")
def get_prediction_logs():
    log_file = "logs/predictions.joblib"

    if not os.path.exists(log_file):
        return JSONResponse(content={"logs": []}, status_code=200)

    try:
        logs = joblib.load(log_file)
        return {"logs": logs}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


    return {"survived": int(prediction)}
    
    
from fastapi.responses import FileResponse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

@app.get("/logs/plot")
def plot_survival_stats():
    log_file = "logs/predictions.joblib"
    if not os.path.exists(log_file):
        return {"error": "No logs found."}

    logs = joblib.load(log_file)
    df = pd.DataFrame(logs)

    # Parse survival label back to binary for plotting
    df['survived'] = df['prediction'].map({'Survived': 1, 'Not Survived': 0})

    if df.empty:
        return {"error": "No prediction data to plot."}

    # Plot: Countplot of Survived vs Not Survived
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x='prediction', palette='Set2')
    plt.title("Survival Count")
    plt.xlabel("Prediction")
    plt.ylabel("Count")
    plt.tight_layout()

    output_path = "logs/survival_plot.png"
    plt.savefig(output_path)
    plt.close()

    return FileResponse(output_path, media_type="image/png", filename="survival_plot.png")


