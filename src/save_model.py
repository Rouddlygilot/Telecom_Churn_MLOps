from pathlib import Path
import joblib

from train import split_data, create_model


def save_model():
    # Séparation des données
    X_train, X_test, y_train, y_test = split_data()

    # Création du modèle Random Forest
    model = create_model("Random Forest", X_train)

    # Entraînement
    model.fit(X_train, y_train)

    # Définition du seuil de décision
    threshold = 0.35

    # Définition du chemin de sauvegarde
    project_dir = Path(__file__).resolve().parent.parent
    model_dir = project_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "random_forest_churn.joblib"

    # Sauvegarde du modèle et du seuil
    joblib.dump(
        {
            "model": model,
            "threshold": threshold
        },
        model_path
    )

    print("\n===== SAUVEGARDE DU MODÈLE =====")
    print("Modèle : Random Forest")
    print("Seuil de décision :", threshold)
    print("Fichier sauvegardé :", model_path)
    print("\nSauvegarde terminée avec succès.")


if __name__ == "__main__":
    save_model()