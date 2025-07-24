from fastapi import FastAPI, Query
import pandas as pd
import joblib

# Load model
model = joblib.load("titanic_model.pkl")

# App init
app = FastAPI()

# Prediction endpoint
@app.get("/predict")
def predict(pclass: int, sex: str, age: float):
    sex_encoded = 0 if sex.lower() == "male" else 1
    input_df = pd.DataFrame([[pclass, sex_encoded, age]], columns=["pclass", "sex", "age"])
    prediction = model.predict(input_df)[0]
    return {"prediction": "Survived" if prediction == 1 else "Did not survive"}

