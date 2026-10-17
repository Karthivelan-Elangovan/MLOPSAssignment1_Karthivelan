from fastapi.testclient import TestClient
from src.api import app


client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model"] == "logistic_regression"


def test_prediction_endpoint():
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
        "cp_1_0": 0,
        "cp_2_0": 0,
        "cp_3_0": 1,
        "cp_4_0": 0,
        "restecg_0_0": 1,
        "restecg_1_0": 0,
        "restecg_2_0": 0,
        "thal_3_0": 1,
        "thal_6_0": 0,
        "thal_7_0": 0,
    }

    response = client.post("/predict", json=patient)

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] in [0, 1]
    assert data["prediction_label"] in [
        "Heart disease detected",
        "No heart disease detected",
    ]
    assert 0 <= data["probability"] <= 1