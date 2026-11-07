from pathlib import Path
import joblib


def load_model():
    project_dir = Path(__file__).resolve().parent.parent
    model_path = project_dir / "models" / "random_forest_churn.joblib"

    saved_data = joblib.load(model_path)

    model = saved_data["model"]
    threshold = saved_data["threshold"]

    print("\n===== VÉRIFICATION DU MODÈLE =====")
    print("Modèle chargé :", type(model.named_steps["classifier"]).__name__)
    print("Seuil de décision :", threshold)
    print("Chargement effectué avec succès.")


if __name__ == "__main__":
    load_model()