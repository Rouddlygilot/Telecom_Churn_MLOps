
import random
import httpx

# Adresse de l'API
API_URL = "http://127.0.0.1:8000/predict"

# Générateur aléatoire reproductible
random.seed(42)

# Catégories possibles
payment_methods = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)"
]

contracts = [
    "Month-to-month",
    "One year",
    "Two year"
]

internet_services = [
    "DSL",
    "Fiber optic"
]

yes_no = ["Yes", "No"]

# Envoi de 30 profils synthétiques
with httpx.Client(timeout=30) as client:

    for i in range(30):

        tenure = random.randint(1, 72)
        monthly_charges = round(random.uniform(20, 120), 2)

        customer = {
            "gender": random.choice(["Male", "Female"]),
            "SeniorCitizen": random.choice([0, 1]),
            "Partner": random.choice(yes_no),
            "Dependents": random.choice(yes_no),
            "tenure": tenure,
            "PhoneService": "Yes",
            "MultipleLines": random.choice(yes_no),
            "InternetService": random.choice(internet_services),
            "OnlineSecurity": random.choice(yes_no),
            "OnlineBackup": random.choice(yes_no),
            "DeviceProtection": random.choice(yes_no),
            "TechSupport": random.choice(yes_no),
            "StreamingTV": random.choice(yes_no),
            "StreamingMovies": random.choice(yes_no),
            "Contract": random.choice(contracts),
            "PaperlessBilling": random.choice(yes_no),
            "PaymentMethod": random.choice(payment_methods),
            "MonthlyCharges": monthly_charges,
            "TotalCharges": round(monthly_charges * tenure, 2)
        }

        try:
            response = client.post(API_URL, json=customer)
            response.raise_for_status()

            result = response.json()

            print(
                f"Client {i + 1:02d}/30 | "
                f"Churn={result['churn']} | "
                f"Probabilité={result['probability']}"
            )

        except httpx.HTTPError as error:
            print(f"Erreur pour le client {i + 1}: {error}")
            break

print("\nEnvoi des profils de test terminé.")
