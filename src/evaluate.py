
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from train import split_data, create_model


def compare_models():

    # Charger une seule fois les données
    X_train, X_test, y_train, y_test = split_data()

    # Liste des modèles à comparer
    models = [
        "Logistic Regression",
        "Random Forest",
        "Gradient Boosting"
    ]

    results = []

    for model_name in models:

        print(f"\nEntraînement : {model_name}")

        # Créer le modèle
        model = create_model(model_name, X_train)

        # Entraîner le modèle
        model.fit(X_train, y_train)

        # Effectuer les prédictions
        y_pred = model.predict(X_test)

        # Calculer les indicateurs
        results.append({
            "Modèle": model_name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(
                y_test, y_pred, zero_division=0
            ),
            "Recall": recall_score(
                y_test, y_pred, zero_division=0
            ),
            "F1-score": f1_score(
                y_test, y_pred, zero_division=0
            )
        })

    # Créer le tableau comparatif
    results_df = pd.DataFrame(results)

    # Trier les modèles selon le Recall
    results_df = results_df.sort_values(
        by="Recall",
        ascending=False
    )

    print("\n===== COMPARAISON DES MODÈLES =====")
    print(results_df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1-score": "{:.4f}".format
        }
    ))

    # Identifier le meilleur modèle selon le Recall
    best_model = results_df.iloc[0]

    print("\n===== MEILLEUR MODÈLE SELON LE RECALL =====")
    print("Modèle :", best_model["Modèle"])
    print("Recall :", round(best_model["Recall"], 4))


if __name__ == "__main__":
    compare_models()

# Optimisation seuil de decision du Random Forest 


import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from train import split_data, create_model


def optimize_threshold():

    X_train, X_test, y_train, y_test = split_data()

    # Créer et entraîner Random Forest
    model = create_model("Random Forest", X_train)
    model.fit(X_train, y_train)

    # Obtenir les probabilités de churn (classe 1)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Seuils à tester
    thresholds = [0.50, 0.45, 0.40, 0.35, 0.30, 0.25]

    results = []

    for threshold in thresholds:

        # Prédictions selon le seuil
        y_pred = (y_proba >= threshold).astype(int)

        results.append({
            "Seuil": threshold,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(
                y_test, y_pred, zero_division=0
            ),
            "Recall": recall_score(
                y_test, y_pred, zero_division=0
            ),
            "F1-score": f1_score(
                y_test, y_pred, zero_division=0
            )
        })

    # Tableau comparatif
    results_df = pd.DataFrame(results)

    print("\n===== OPTIMISATION DU SEUIL DE DÉCISION =====")

    print(results_df.to_string(
        index=False,
        formatters={
            "Seuil": "{:.2f}".format,
            "Accuracy": "{:.4f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1-score": "{:.4f}".format
        }
    ))


if __name__ == "__main__":
    optimize_threshold()

# Matrice de confusion

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from train import split_data, create_model


def evaluate_random_forest():

    X_train, X_test, y_train, y_test = split_data()

    # Créer et entraîner Random Forest
    model = create_model("Random Forest", X_train)
    model.fit(X_train, y_train)

    # Obtenir les probabilités de churn
    y_proba = model.predict_proba(X_test)[:, 1]

    # Appliquer le seuil retenu
    threshold = 0.35
    y_pred = (y_proba >= threshold).astype(int)

    # Calculer les indicateurs
    print("\n===== ÉVALUATION DE RANDOM FOREST =====")
    print("Seuil de décision :", threshold)

    print("\nAccuracy :", round(
        accuracy_score(y_test, y_pred), 4
    ))

    print("Precision :", round(
        precision_score(y_test, y_pred), 4
    ))

    print("Recall :", round(
        recall_score(y_test, y_pred), 4
    ))

    print("F1-score :", round(
        f1_score(y_test, y_pred), 4
    ))

    # Rapport de classification
    print("\n===== RAPPORT DE CLASSIFICATION =====")

    print(classification_report(
        y_test,
        y_pred,
        target_names=["No (client fidèle)", "Yes (client parti)"],
        zero_division=0
    ))

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)

    print("\n===== MATRICE DE CONFUSION =====")
    print(cm)

    print("\nDétail de la matrice :")
    print("Vrais négatifs :", cm[0, 0])
    print("Faux positifs :", cm[0, 1])
    print("Faux négatifs :", cm[1, 0])
    print("Vrais positifs :", cm[1, 1])


if __name__ == "__main__":
    evaluate_random_forest()
