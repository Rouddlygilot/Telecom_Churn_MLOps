
from pathlib import Path
import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

# Importation de la fonction de monitoring
from src.monitoring import log_prediction


# Initialisation de l'API
app = FastAPI(
    title="Telecom Customer Churn Prediction API",
    description="API de prédiction de résiliation des clients télécoms",
    version="1.1.0"
)


# Chargement du modèle sauvegardé
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "random_forest_churn.joblib"

saved_data = joblib.load(MODEL_PATH)

model = saved_data["model"]
threshold = saved_data["threshold"]


# Structure des données reçues
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


# Route d'accueil
@app.get("/")
def home():
    return {
        "message": "Bienvenue dans l'API de prédiction du Churn télécom"
    }


# Route de prédiction
@app.post("/predict")
def predict_churn(customer: CustomerData):

    # Transformer les données reçues en DataFrame
    customer_df = pd.DataFrame([customer.model_dump()])

    # Calculer la probabilité de résiliation
    probability = float(model.predict_proba(customer_df)[0][1])

    # Déterminer la classe selon le seuil
    prediction = int(probability >= threshold)

    churn = "Yes" if prediction == 1 else "No"

    # Enregistrer la prédiction dans le fichier CSV
    log_prediction(
        prediction=prediction,
        churn=churn,
        probability=round(probability, 4),
        threshold=threshold
    )

    # Retourner le résultat à l'utilisateur
    return {
        "prediction": prediction,
        "churn": churn,
        "probability": round(probability, 4),
        "threshold": threshold
    }
