# Telecom Churn – Projet MLOps

## 1. Présentation du projet

Ce projet consiste à transformer un modèle de Machine Learning de prédiction du désabonnement des clients télécoms (*Customer Churn*) en une application déployable, testable, monitorée et maintenable.

L'objectif est de mettre en place une chaîne d'industrialisation permettant d'exposer un modèle de Machine Learning à travers une API, de le déployer avec Docker et de suivre ses prédictions grâce à un tableau de bord.

## 2. Technologies utilisées

- Python
- Pandas et NumPy
- Scikit-learn
- Joblib
- FastAPI
- Uvicorn
- Docker
- Streamlit
- Matplotlib
- Pytest

## 3. Architecture du projet

```text
Telecom_Churn_MLOps/
│
├── data/
│   └── ...
│
├── models/
│   └── random_forest_churn.joblib
│
├── src/
│   ├── api.py
│   ├── data.py
│   ├── features.py
│   └── monitoring.py
│
├── tests/
│   ├── test_api.py
│   ├── test_data.py
│   └── test_features.py
│
├── logs/
│   └── predictions.csv
│
├── dashboard.py
├── Dockerfile
├── requirements.txt
└── README.md
```

## 4. Installation

### 4.1 Cloner ou récupérer le projet

Placer le projet dans un dossier de travail local.

### 4.2 Créer un environnement virtuel

```bash
python -m venv envtelcom_churn
```

### 4.3 Activer l'environnement virtuel sous Windows

```cmd
envtelcom_churn\Scripts\activate
```

### 4.4 Installer les dépendances

```bash
pip install -r requirements.txt
```

## 5. Lancement de l'API FastAPI

À la racine du projet, exécuter :

```bash
uvicorn src.api:app --reload
```

L'API est accessible à l'adresse :

http://127.0.0.1:8000

La documentation interactive Swagger est accessible à :

http://127.0.0.1:8000/docs

### Endpoint de prédiction

- Méthode : `POST`
- Route : `/predict`
- Fonction : recevoir les caractéristiques d'un client et retourner une prédiction de désabonnement.

La réponse contient notamment :

- `prediction` : classe prédite (0 ou 1).
- `churn` : résultat (`Yes` ou `No`).
- `probability` : probabilité estimée de churn.
- `threshold` : seuil de décision utilisé.

## 6. Déploiement avec Docker

Construire l'image Docker :

```bash
docker build -t telecom-churn-api .
```

Lancer le conteneur :

```bash
docker run -d -p 8000:8000 -v "%cd%\logs:/app/logs" --name telecom-churn-container telecom-churn-api
```

Vérifier son fonctionnement :

```bash
docker ps
```

## 7. Monitoring des prédictions

Chaque prédiction effectuée par l'API est enregistrée dans :

```text
logs/predictions.csv
```

Les informations enregistrées comprennent :

- La date et l'heure de la prédiction.
- La classe prédite.
- Le résultat churn.
- La probabilité estimée.
- Le seuil de décision.

Ce mécanisme permet de conserver un historique des prédictions réalisées.

## 8. Tableau de bord Streamlit

Pour démarrer le tableau de bord, exécuter :

```bash
streamlit run dashboard.py
```

Le tableau de bord permet de visualiser :

- Le nombre total de prédictions.
- Le nombre de clients classés Churn Yes et Churn No.
- Le taux de churn prédit.
- La probabilité moyenne de churn.
- La distribution des probabilités.
- L'évolution du volume des prédictions.
- L'historique des prédictions.

Le tableau de bord s'actualise automatiquement toutes les 10 secondes.

## 9. Tests automatisés

Les tests sont réalisés avec Pytest.

Exécuter tous les tests :

```bash
pytest -v
```

Les tests couvrent trois domaines :

- API : fonctionnement des routes et validation des requêtes.
- Données : chargement, dimensions, colonnes et qualité des données.
- Variables : séparation des caractéristiques, encodage et prétraitement.

Résultat obtenu lors de la validation du projet :

**13 tests réussis sur 13.**

## 10. Conclusion

Ce projet met en œuvre une chaîne d'industrialisation d'un modèle de Machine Learning appliqué à la prédiction du désabonnement des clients télécoms.

Il combine le développement d'une API, la conteneurisation avec Docker, l'enregistrement des prédictions, leur visualisation dans un tableau de bord et la validation du fonctionnement grâce aux tests automatisés.

L'ensemble constitue une première application MLOps fonctionnelle, avec des possibilités d'amélioration futures telles que le suivi des performances réelles du modèle et la détection du data drift.