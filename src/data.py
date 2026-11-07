
import pandas as pd
from pathlib import Path


def load_data():
    """
    Charge le dataset Telco Customer Churn
    et convertit TotalCharges en données numériques.
    """

    # 1. Déterminer le chemin du projet
    project_dir = Path(__file__).resolve().parent.parent

    # 2. Déterminer le chemin du fichier CSV
    data_path = project_dir / "data" / "Telco-Customer-Churn.csv"

    # 3. Vérifier que le fichier existe
    if not data_path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {data_path}"
        )

    # 4. Charger le fichier CSV
    df = pd.read_csv(data_path)

    # 5. Nettoyer et convertir TotalCharges en numérique
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].astype(str).str.strip(),
        errors="coerce"
    )

    # 6. Retourner le DataFrame
    return df


if __name__ == "__main__":

    # Charger les données
    df = load_data()

    print("\n===== DIMENSIONS DU DATASET =====")
    print(df.shape)

    print("\n===== LES 5 PREMIÈRES LIGNES =====")
    print(df.head())

    print("\n===== INFORMATIONS SUR LES VARIABLES =====")
    df.info()

    print("\n===== VALEURS MANQUANTES APRÈS CONVERSION =====")
    print(df.isnull().sum())

    print("\n===== NOMBRE TOTAL DE VALEURS MANQUANTES =====")
    print(df.isnull().sum().sum())

    print("\n===== RÉPARTITION DU CHURN =====")
    print(df["Churn"].value_counts())

    print("\n===== VÉRIFICATION DES DOUBLONS =====")
    print(
        "Nombre de clients en double :",
        df["customerID"].duplicated().sum()
    )

    print("\n===== VÉRIFICATION DE TOTALCHARGES =====")
    print(
        "Nombre de valeurs manquantes dans TotalCharges :",
        df["TotalCharges"].isnull().sum()
    )
    # Remplacer les valeurs manquantes de TotalCharges par 0
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    
    print("\n===== STATISTIQUES DES VARIABLES NUMÉRIQUES =====")
    print(
        df[["tenure", "MonthlyCharges", "TotalCharges"]].describe()
    )
    print("\n===== EXAMEN DES VALEURS MANQUANTES =====")

    clients_manquants = df[df["TotalCharges"].isnull()]

    print(clients_manquants[
        ["customerID", "tenure", "MonthlyCharges",
         "TotalCharges", "Churn"]
    ])

    print("\n===== ANCIENNETÉ DES CLIENTS CONCERNÉS =====")
    print(clients_manquants["tenure"].value_counts())

    print("\n===== BILAN FINAL DU NETTOYAGE =====")

    print("Dimensions du dataset :", df.shape)

    print("\nValeurs manquantes par colonne :")
    print(df.isnull().sum())

    print("\nNombre total de valeurs manquantes :",
          df.isnull().sum().sum())

    print("\nNombre de doublons :",
          df["customerID"].duplicated().sum())

    print("\nType de TotalCharges :",
          df["TotalCharges"].dtype)

    print("\n===== IDENTIFICATION DES VARIABLES =====")

    # Identifier les variables numériques
    variables_numeriques = df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    # Identifier les variables catégorielles
    variables_categorielles = df.select_dtypes(
        include=["object", "str"]
    ).columns.tolist()

    print("\nVariables numériques :")
    print(variables_numeriques)

    print("\nVariables catégorielles :")
    print(variables_categorielles)

    print("\nNombre de variables numériques :",
          len(variables_numeriques))

    print("Nombre de variables catégorielles :",
          len(variables_categorielles))

    print("\n===== ANALYSE DE LA VARIABLE CIBLE =====")

    # Effectifs de chaque classe
    print("\nEffectifs :")
    print(df["Churn"].value_counts())

    # Pourcentages de chaque classe
    print("\nRépartition en pourcentage :")
    print(df["Churn"].value_counts(normalize=True) * 100)

    # Nombre de classes
    print("\nNombre de classes :",
          df["Churn"].nunique())

    # Modalités de la variable cible
    print("\nModalités de Churn :",
          df["Churn"].unique())