
from pathlib import Path

import numpy as np
import pandas as pd


# ==========================================
# 1. CONFIGURATION
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

TRAIN_FILE = PROJECT_DIR / "data" / "Telco-Customer-Churn.csv"
NEW_DATA_FILE = PROJECT_DIR / "logs" / "prediction_inputs.csv"

REPORT_DIR = PROJECT_DIR / "reports"
REPORT_FILE = REPORT_DIR / "data_drift_report.csv"

MIN_NEW_OBSERVATIONS = 30

NUMERIC_COLUMNS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

CATEGORICAL_COLUMNS = [
    "gender",
    "Partner",
    "Dependents",
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
    "PaymentMethod"
]


# ==========================================
# 2. CALCUL DU POPULATION STABILITY INDEX
# ==========================================

def calculate_psi(reference, current, categorical=False):
    """
    Calcule le PSI entre les données de référence
    et les nouvelles données.
    """

    reference = pd.Series(reference).dropna()
    current = pd.Series(current).dropna()

    if len(reference) == 0 or len(current) == 0:
        return np.nan

    if categorical:
        categories = sorted(
            set(reference.astype(str).unique())
            | set(current.astype(str).unique())
        )

        reference = reference.astype(str)
        current = current.astype(str)

        reference_counts = (
            reference.value_counts()
            .reindex(categories, fill_value=0)
        )

        current_counts = (
            current.value_counts()
            .reindex(categories, fill_value=0)
        )

    else:
        # Création de classes à partir des quantiles
        quantiles = np.linspace(0, 1, 6)

        bins = np.unique(
            reference.quantile(quantiles).to_numpy()
        )

        if len(bins) < 2:
            return 0.0

        bins[0] = -np.inf
        bins[-1] = np.inf

        reference_counts = pd.cut(
            reference,
            bins=bins,
            include_lowest=True
        ).value_counts(sort=False)

        current_counts = pd.cut(
            current,
            bins=bins,
            include_lowest=True
        ).value_counts(sort=False)

    # Conversion en proportions
    epsilon = 0.0001

    reference_pct = (
        reference_counts / reference_counts.sum()
    ).clip(lower=epsilon)

    current_pct = (
        current_counts / current_counts.sum()
    ).clip(lower=epsilon)

    # Calcul du PSI
    psi = np.sum(
        (current_pct - reference_pct)
        * np.log(current_pct / reference_pct)
    )

    return float(psi)


# ==========================================
# 3. CLASSIFICATION DU NIVEAU DE DRIFT
# ==========================================

def classify_drift(psi):

    if pd.isna(psi):
        return "Non calculable"

    if psi < 0.10:
        return "Faible"

    elif psi < 0.20:
        return "Modéré"

    else:
        return "Important"


# ==========================================
# 4. COMPARAISON DES DONNEES
# ==========================================

def detect_data_drift():

    print("\n========== ANALYSE DU DATA DRIFT ==========")

    # Chargement des données
    train = pd.read_csv(TRAIN_FILE)
    new_data = pd.read_csv(NEW_DATA_FILE)

    # Conversion des variables numériques
    for column in NUMERIC_COLUMNS:
        train[column] = pd.to_numeric(
            train[column], errors="coerce"
        )
        new_data[column] = pd.to_numeric(
            new_data[column], errors="coerce"
        )

    print("Données de référence :", train.shape)
    print("Nouvelles données :", new_data.shape)

    # Vérification du nombre d'observations
    if len(new_data) < MIN_NEW_OBSERVATIONS:

        print(
            "\nATTENTION : nombre insuffisant de nouvelles "
            "observations pour une analyse fiable."
        )

        print(
            f"Observations disponibles : {len(new_data)}"
        )

        print(
            f"Minimum requis : {MIN_NEW_OBSERVATIONS}"
        )

        print(
            "\nAucune conclusion de Data Drift ne sera produite."
        )

        return None

    # Comparaison des variables
    results = []

    for column in NUMERIC_COLUMNS + CATEGORICAL_COLUMNS:

        is_categorical = column in CATEGORICAL_COLUMNS

        psi = calculate_psi(
            train[column],
            new_data[column],
            categorical=is_categorical
        )

        results.append({
            "variable": column,
            "type": "Catégorielle" if is_categorical else "Numérique",
            "PSI": round(psi, 4),
            "niveau_drift": classify_drift(psi)
        })

    # Création du rapport
    report = pd.DataFrame(results)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report.to_csv(REPORT_FILE, index=False, encoding="utf-8")

    print("\n========== RAPPORT DE DATA DRIFT ==========")
    print(report.to_string(index=False))

    print("\nRapport enregistré dans :", REPORT_FILE)

    return report


# ==========================================
# 5. EXECUTION
# ==========================================

if __name__ == "__main__":
    detect_data_drift()
