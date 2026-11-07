
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def separate_features_target(df):
    """
    Sépare les variables explicatives X
    de la variable cible y.
    """
    X = df.drop(columns=["customerID", "Churn"])

    y = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    return X, y


def create_preprocessor(X):
    """
    Crée le pipeline de prétraitement
    des variables numériques et catégorielles.
    """

    numeric_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "str"]
    ).columns.tolist()

    # Pipeline numérique
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    # Pipeline catégoriel
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    # Regroupement des transformations
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor


if __name__ == "__main__":

    from data import load_data

    df = load_data()

    X, y = separate_features_target(df)

    preprocessor = create_preprocessor(X)

    X_transformed = preprocessor.fit_transform(X)

    print("\n===== PRÉTRAITEMENT DES DONNÉES =====")

    print("\nDimensions initiales de X :")
    print(X.shape)

    print("\nDimensions après transformation :")
    print(X_transformed.shape)

    print("\nValeurs manquantes dans X :")
    print(X.isnull().sum().sum())

    print("\nValeurs manquantes après transformation :")
    print(pd.isna(X_transformed).sum())

    print("\nPrétraitement terminé avec succès.")
