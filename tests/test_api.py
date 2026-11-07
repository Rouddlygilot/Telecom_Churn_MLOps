
from fastapi.testclient import TestClient

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api import app

client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert "Bienvenue" in response.json()["message"]


def test_predict():
    customer = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.1,
        "TotalCharges": 1071.7
    }

    response = client.post("/predict", json=customer)

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] in [0, 1]
    assert result["churn"] in ["Yes", "No"]
    assert 0 <= result["probability"] <= 1
    assert result["threshold"] == 0.35


def test_predict_missing_field():
    customer = {
        "gender": "Male"
    }

    response = client.post("/predict", json=customer)

    assert response.status_code == 422
