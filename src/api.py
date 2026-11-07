
from pathlib import Path
import logging
import time

import joblib
import pandas as pd

from fastapi import FastAPI, Request
from pydantic import BaseModel

from src.monitoring import log_prediction, log_prediction_input


# ==========================================
# 1. CONFIGURATION DES DOSSIERS ET DES LOGS
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

LOG_DIR = PROJECT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "api.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ],
    force=True
)

logger = logging.getLogger("telecom_api")


# ==========================================
# 2. INITIALISATION DE L'API
# ==========================================

app = FastAPI(
    title="Telecom Customer Churn Prediction API",
    description="API de prédiction de résiliation des clients télécoms",
    version="1.3.0"
)


# ==========================================
# 3. MIDDLEWARE DE SURVEILLANCE
# ==========================================

@app.middleware("http")
async def monitor_requests(request: Request, call_next):

    start_time = time.perf_counter()

    try:
        response = await call_next(request)

        duration = (time.perf_counter() - start_time) * 1000

        logger.info(
            "Méthode=%s | URL=%s | Statut=%s | Temps=%.2f ms",
            request.method,
            request.url.path,
            response.status_code,
            duration
        )

        return response

    except Exception:
        duration = (time.perf_counter() - start_time) * 1000

        logger.exception(
            "Erreur API | Méthode=%s | URL=%s | Temps=%.2f ms",
            request.method,
            request.url.path,
            duration
        )

        raise


# ==========================================
# 4. CHARGEMENT DU MODELE
# ==========================================

MODEL_PATH = PROJECT_DIR / "models" / "random_forest_churn.joblib"

saved_data = joblib.load(MODEL_PATH)

model = saved_data["model"]
threshold = saved_data["threshold"]


# ==========================================
# 5. STRUCTURE DES DONNEES RECUES
# ==========================================

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


# ==========================================
# 6. ROUTE D'ACCUEIL
# ==========================================

@app.get("/")
def home():

    logger.info("Accès à la route d'accueil")

    return {
        "message": "Bienvenue dans l'API de prédiction du Churn télécom"
    }


# ==========================================
# 7. ROUTE DE PREDICTION
# ==========================================

@app.post("/predict")
def predict_churn(customer: CustomerData):

    # Récupérer les caractéristiques du client
    customer_data = customer.model_dump()

    # Enregistrer les caractéristiques reçues
    log_prediction_input(customer_data)

    # Transformer les données en DataFrame
    customer_df = pd.DataFrame([customer_data])

    # Calculer la probabilité de résiliation
    probability = float(model.predict_proba(customer_df)[0][1])

    # Déterminer la classe selon le seuil
    prediction = int(probability >= threshold)

    churn = "Yes" if prediction == 1 else "No"

    # Enregistrer le résultat de la prédiction
    log_prediction(
        prediction=prediction,
        churn=churn,
        probability=round(probability, 4),
        threshold=threshold
    )

    logger.info(
        "Prédiction réalisée | Churn=%s | Probabilité=%.4f | Seuil=%.2f",
        churn,
        probability,
        threshold
    )

    return {
        "prediction": prediction,
        "churn": churn,
        "probability": round(probability, 4),
        "threshold": threshold
    }
