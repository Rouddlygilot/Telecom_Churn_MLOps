
from pathlib import Path
from datetime import datetime, timezone
import csv

PROJECT_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "predictions.csv"
INPUT_LOG_FILE = LOG_DIR / "prediction_inputs.csv"

FIELDS = [
    "timestamp_utc",
    "prediction",
    "churn",
    "probability",
    "threshold"
]

INPUT_FIELDS = [
    "timestamp_utc",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges"
]


def log_prediction(prediction, churn, probability, threshold):
    """Enregistre les résultats des prédictions."""
    file_exists = LOG_FILE.exists()

    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)

        if not file_exists or LOG_FILE.stat().st_size == 0:
            writer.writeheader()

        writer.writerow({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "prediction": prediction,
            "churn": churn,
            "probability": probability,
            "threshold": threshold
        })

    return str(LOG_FILE)


def log_prediction_input(customer_data):
    """Enregistre les caractéristiques reçues par l'API."""
    file_exists = INPUT_LOG_FILE.exists()

    with open(INPUT_LOG_FILE, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=INPUT_FIELDS,
            extrasaction="ignore"
        )

        if not file_exists or INPUT_LOG_FILE.stat().st_size == 0:
            writer.writeheader()

        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **customer_data
        }

        writer.writerow(row)

    return str(INPUT_LOG_FILE)
