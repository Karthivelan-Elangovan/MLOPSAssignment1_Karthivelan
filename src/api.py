from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "logistic_regression_model.joblib"


# Load trained model
model = joblib.load(MODEL_PATH)


# Create FastAPI application
app = FastAPI(
    title="Heart Disease Prediction API",
    description="API for predicting heart disease using a trained Logistic Regression model",
    version="1.0.0",
)


class PatientData(BaseModel):
    age: float
    sex: float
    trestbps: float
    chol: float
    fbs: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float

    cp_1_0: float
    cp_2_0: float
    cp_3_0: float
    cp_4_0: float

    restecg_0_0: float
    restecg_1_0: float
    restecg_2_0: float

    thal_3_0: float
    thal_6_0: float
    thal_7_0: float


@app.get("/")
def root():
    return {
        "message": "Heart Disease Prediction API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "logistic_regression"
    }


@app.post("/predict")
def predict(patient: PatientData):

    patient_dict = patient.model_dump()

    # Convert API field names back to model column names
    patient_data = {
        "age": patient_dict["age"],
        "sex": patient_dict["sex"],
        "trestbps": patient_dict["trestbps"],
        "chol": patient_dict["chol"],
        "fbs": patient_dict["fbs"],
        "thalach": patient_dict["thalach"],
        "exang": patient_dict["exang"],
        "oldpeak": patient_dict["oldpeak"],
        "slope": patient_dict["slope"],
        "ca": patient_dict["ca"],

        "cp_1.0": patient_dict["cp_1_0"],
        "cp_2.0": patient_dict["cp_2_0"],
        "cp_3.0": patient_dict["cp_3_0"],
        "cp_4.0": patient_dict["cp_4_0"],

        "restecg_0.0": patient_dict["restecg_0_0"],
        "restecg_1.0": patient_dict["restecg_1_0"],
        "restecg_2.0": patient_dict["restecg_2_0"],

        "thal_3.0": patient_dict["thal_3_0"],
        "thal_6.0": patient_dict["thal_6_0"],
        "thal_7.0": patient_dict["thal_7_0"],
    }

    input_df = pd.DataFrame([patient_data])

    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0][1])

    label = (
        "Heart disease detected"
        if prediction == 1
        else "No heart disease detected"
    )

    return {
        "prediction": prediction,
        "prediction_label": label,
        "probability": round(probability, 4)
    }