from pathlib import Path

import joblib
import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "logistic_regression_model.joblib"


def load_model():
    """Load the serialized Logistic Regression pipeline."""
    model = joblib.load(MODEL_PATH)
    return model


def prepare_input():
    """Create a sample patient input with the required model features."""

    patient = {
        "age": 55,
        "sex": 1,
        "trestbps": 140,
        "chol": 250,
        "fbs": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 1.0,
        "slope": 2,
        "ca": 0,
        "cp_1.0": 0,
        "cp_2.0": 0,
        "cp_3.0": 1,
        "cp_4.0": 0,
        "restecg_0.0": 1,
        "restecg_1.0": 0,
        "restecg_2.0": 0,
        "thal_3.0": 1,
        "thal_6.0": 0,
        "thal_7.0": 0,
    }

    return pd.DataFrame([patient])


def predict_heart_disease(model, patient_data):
    """Generate prediction and probability."""
    prediction = model.predict(patient_data)[0]
    probability = model.predict_proba(patient_data)[0][1]

    return prediction, probability


if __name__ == "__main__":
    print("Loading serialized model...")

    model = load_model()

    patient_data = prepare_input()

    print("Model loaded successfully.")
    print("\nPatient input:")
    print(patient_data.to_string(index=False))

    prediction, probability = predict_heart_disease(
        model,
        patient_data
    )

    print("\nPrediction Result")
    print("=================")

    if prediction == 1:
        print("Prediction: Heart disease detected")
    else:
        print("Prediction: No heart disease detected")

    print(f"Probability of heart disease: {probability:.2%}")