
from pathlib import Path
from datetime import datetime, timezone
import csv

# Chemin du projet
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Dossier et fichier de suivi des prédictions
LOG_DIR = PROJECT_DIR / "logs"
LOG_FILE = LOG_DIR / "predictions.csv"

# Créer le dossier logs s'il n'existe pas
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Colonnes du fichier CSV
FIELDS = [
    "timestamp_utc",
    "prediction",
    "churn",
    "probability",
    "threshold"
]


def log_prediction(prediction, churn, probability, threshold):
    """Enregistre chaque prédiction dans un fichier CSV."""

    file_exists = LOG_FILE.exists()

    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)

        # Ajouter les en-têtes si le fichier est nouveau ou vide
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
