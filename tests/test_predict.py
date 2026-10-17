from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "logistic_regression_model.joblib"


def test_model_file_exists():
    """Verify that the serialized model exists."""
    assert MODEL_PATH.exists()


def test_model_can_be_loaded():
    """Verify that the serialized model can be loaded."""
    model = joblib.load(MODEL_PATH)

    assert model is not None


def test_model_prediction():
    """Verify that the model can generate a prediction."""

    model = joblib.load(MODEL_PATH)

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

    patient_data = pd.DataFrame([patient])

    prediction = model.predict(patient_data)

    assert prediction[0] in [0, 1]