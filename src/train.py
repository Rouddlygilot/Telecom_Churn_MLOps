from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from data import load_data
from features import separate_features_target, create_preprocessor


def split_data():
    """
    Charge les données et les divise
    en ensembles d'entraînement et de test.
    """

    df = load_data()
    X, y = separate_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def create_model(model_type, X_train):
    """
    Crée un pipeline avec le prétraitement
    et l'algorithme choisi.
    """

    preprocessor = create_preprocessor(X_train)

    if model_type == "Logistic Regression":

        classifier = LogisticRegression(
            max_iter=1000,
            random_state=42
        )

    elif model_type == "Random Forest":

        classifier = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        )

    elif model_type == "Gradient Boosting":

        classifier = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        )

    else:
        raise ValueError("Modèle non reconnu.")

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ]
    )

    return model


def train_model():

    X_train, X_test, y_train, y_test = split_data()

    model = create_model(
        "Logistic Regression",
        X_train
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    return model, accuracy


if __name__ == "__main__":

    model, accuracy = train_model()

    print("\n===== RÉSULTATS DU PREMIER MODÈLE =====")

    print("Modèle : Régression logistique")

    print(
        f"Accuracy : {accuracy:.4f}"
    )