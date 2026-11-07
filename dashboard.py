
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_autorefresh import st_autorefresh


# ==========================================
# 1. CONFIGURATION DU TABLEAU DE BORD
# ==========================================

st.set_page_config(
    page_title="Telecom Churn Monitoring",
    page_icon="📊",
    layout="wide"
)

# Actualisation automatique toutes les 10 secondes
st_autorefresh(
    interval=10 * 1000,
    key="dashboard_refresh"
)

st.title("📊 Telecom Churn Monitoring")

st.markdown(
    "Tableau de bord de suivi des prédictions "
    "du modèle Random Forest."
)

st.caption("Actualisation automatique toutes les 10 secondes.")


# ==========================================
# 2. CHARGEMENT DES DONNEES
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent
LOG_FILE = PROJECT_DIR / "logs" / "predictions.csv"

if not LOG_FILE.exists():
    st.warning("Aucun fichier de monitoring trouvé.")
    st.info("Effectuez d'abord une prédiction avec l'API.")
    st.stop()

try:
    df = pd.read_csv(LOG_FILE)

except (pd.errors.EmptyDataError, PermissionError, OSError) as error:
    st.error(f"Impossible de lire le fichier de monitoring : {error}")
    st.stop()

if df.empty:
    st.warning("Le fichier de monitoring ne contient aucune prédiction.")
    st.stop()


# Vérification des colonnes nécessaires
required_columns = [
    "timestamp_utc",
    "prediction",
    "churn",
    "probability",
    "threshold"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error(
        f"Colonnes manquantes dans le fichier CSV : {missing_columns}"
    )
    st.stop()


# ==========================================
# 3. PREPARATION DES DONNEES
# ==========================================

df["timestamp_utc"] = pd.to_datetime(
    df["timestamp_utc"],
    errors="coerce",
    utc=True
)

df["prediction"] = pd.to_numeric(
    df["prediction"],
    errors="coerce"
)

df["probability"] = pd.to_numeric(
    df["probability"],
    errors="coerce"
)

df["threshold"] = pd.to_numeric(
    df["threshold"],
    errors="coerce"
)

# Suppression des lignes invalides
df = df.dropna(
    subset=[
        "timestamp_utc",
        "prediction",
        "probability",
        "threshold"
    ]
)

if df.empty:
    st.warning("Aucune donnée valide à afficher.")
    st.stop()

# Conserver uniquement les classes 0 et 1
df = df[df["prediction"].isin([0, 1])].copy()

if df.empty:
    st.warning("Aucune prédiction valide (0 ou 1) trouvée.")
    st.stop()

# Date de dernière prédiction
last_prediction_date = df["timestamp_utc"].max()


# ==========================================
# 4. INDICATEURS CLES (KPI)
# ==========================================

total_predictions = len(df)

churn_yes = (df["prediction"] == 1).sum()
churn_no = (df["prediction"] == 0).sum()

churn_rate = (
    churn_yes / total_predictions * 100
)

average_probability = (
    df["probability"].mean() * 100
)

st.subheader("Indicateurs clés")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total des prédictions",
    total_predictions
)

col2.metric(
    "Clients Churn Yes",
    churn_yes
)

col3.metric(
    "Clients Churn No",
    churn_no
)

col4.metric(
    "Taux de churn prédit",
    f"{churn_rate:.2f}%"
)

st.metric(
    "Probabilité moyenne de churn",
    f"{average_probability:.2f}%"
)

st.caption(
    "Dernière prédiction enregistrée (UTC) : "
    f"{last_prediction_date.strftime('%d/%m/%Y %H:%M:%S')}"
)


# ==========================================
# 5. REPARTITION DES PREDICTIONS
# ==========================================

st.subheader("Répartition des prédictions")

fig, ax = plt.subplots()

labels = []
values = []

if churn_yes > 0:
    labels.append("Churn Yes")
    values.append(churn_yes)

if churn_no > 0:
    labels.append("Churn No")
    values.append(churn_no)

ax.pie(
    values,
    labels=labels,
    autopct="%1.1f%%",
    startangle=90
)

ax.axis("equal")

st.pyplot(fig)
plt.close(fig)


# ==========================================
# 6. DISTRIBUTION DES PROBABILITES
# ==========================================

st.subheader("Distribution des probabilités de churn")

fig, ax = plt.subplots()

ax.hist(
    df["probability"],
    bins=10,
    edgecolor="black"
)

ax.set_xlabel("Probabilité de churn")
ax.set_ylabel("Nombre de prédictions")
ax.set_title("Distribution des probabilités")

ax.set_xlim(0, 1)

st.pyplot(fig)
plt.close(fig)


# ==========================================
# 7. EVOLUTION DES PREDICTIONS
# ==========================================

st.subheader("Évolution des prédictions dans le temps")

df_time = df.sort_values("timestamp_utc").copy()

df_time["nombre_predictions"] = range(
    1,
    len(df_time) + 1
)

fig, ax = plt.subplots()

ax.plot(
    df_time["timestamp_utc"],
    df_time["nombre_predictions"],
    marker="o"
)

ax.set_xlabel("Date et heure (UTC)")
ax.set_ylabel("Nombre cumulé de prédictions")
ax.set_title("Volume cumulé des prédictions")

plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)
plt.close(fig)


# ==========================================
# 8. HISTORIQUE DES PREDICTIONS
# ==========================================

st.subheader("Historique des prédictions")

df_display = df.sort_values(
    "timestamp_utc",
    ascending=False
).copy()

# Format lisible de la date
df_display["timestamp_utc"] = (
    df_display["timestamp_utc"]
    .dt.strftime("%d/%m/%Y %H:%M:%S UTC")
)

# Affichage du tableau
st.dataframe(
    df_display,
    use_container_width=True,
    hide_index=True
)


# ==========================================
# 9. INFORMATIONS GENERALES
# ==========================================

st.divider()

st.caption(
    "Source des données : logs/predictions.csv | "
    "Monitoring des prédictions du modèle Telecom Churn."
)

st.caption(
    "Les indicateurs représentent les prédictions enregistrées "
    "et non les résiliations réellement observées."
)
