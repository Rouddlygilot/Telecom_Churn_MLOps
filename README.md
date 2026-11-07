# Telecom Churn MLOps

## 1. Présentation du projet

Ce projet consiste à industrialiser un modèle de Machine Learning destiné à prédire le désabonnement (*Customer Churn*) des clients d'une entreprise de télécommunications.

L'objectif est de transformer un modèle prédictif en une application exploitable, déployable et surveillable, en appliquant les bonnes pratiques MLOps.

Le projet couvre notamment la préparation des données, l'entraînement et l'évaluation du modèle, le développement d'une API, la conteneurisation, les tests automatisés et la surveillance des prédictions.

## 2. Objectifs du projet

- Préparer et traiter les données des clients télécom.
- Développer un modèle de classification pour prédire le désabonnement.
- Évaluer le modèle à l'aide de tests et de métriques adaptées.
- Exposer le modèle à travers une API REST avec FastAPI.
- Conteneuriser l'application avec Docker.
- Automatiser les tests avec Pytest et GitHub Actions.
- Enregistrer et surveiller les prédictions.
- Visualiser les résultats à travers un tableau de bord Streamlit.
- Détecter les changements dans la distribution des données (*Data Drift*).

## 3. Technologies utilisées

| Technologie | Utilisation |
|---|---|
| Python | Développement du projet |
| Pandas et NumPy | Manipulation et traitement des données |
| Scikit-learn | Prétraitement et Machine Learning |
| Joblib | Sauvegarde et chargement du modèle |
| FastAPI | Développement de l'API REST |
| Uvicorn | Exécution de l'API |
| Docker | Conteneurisation de l'application |
| Streamlit | Tableau de bord de surveillance |
| Matplotlib | Visualisation des données |
| Pytest | Tests automatisés |
| Git et GitHub | Gestion du code source |
| GitHub Actions | Intégration continue (CI) |

## 4. Fonctionnalités principales

### 4.1 Préparation et entraînement du modèle
- Chargement du jeu de données télécom.
- Nettoyage et préparation des variables.
- Prétraitement des données.
- Entraînement du modèle de classification.
- Évaluation du modèle.
- Sauvegarde du modèle entraîné.

### 4.2 API de prédiction
Une API développée avec FastAPI permet de transmettre les caractéristiques d'un client et d'obtenir une prédiction de désabonnement.

Documentation interactive de l'API :

http://localhost:8000/docs

### 4.3 Déploiement avec Docker
L'application est conteneurisée avec Docker afin de faciliter son exécution dans un environnement isolé et reproductible.

### 4.4 Tests automatisés
Les tests sont réalisés avec Pytest afin de vérifier notamment :
- Le chargement des données.
- La structure du jeu de données.
- Le fonctionnement du prétraitement.
- Le fonctionnement des endpoints de l'API.

Le projet comporte 13 tests automatisés.

### 4.5 Surveillance des prédictions
Les prédictions réalisées par l'API sont enregistrées dans des fichiers CSV. Les journaux permettent également de suivre les requêtes et les erreurs de l'API.

### 4.6 Tableau de bord
Un tableau de bord Streamlit permet de visualiser :
- Le nombre total de prédictions.
- La répartition des prédictions de churn et de non-churn.
- Le taux de churn prédit.
- La probabilité moyenne de désabonnement.
- L'historique des prédictions.
- Les indicateurs de Data Drift.

### 4.7 Détection du Data Drift
Un module de surveillance compare les nouvelles données aux données de référence à l'aide du Population Stability Index (PSI).

Les résultats sont classés en trois niveaux :
- Faible : PSI inférieur à 0,10.
- Modéré : PSI compris entre 0,10 et 0,20.
- Important : PSI supérieur ou égal à 0,20.

**Remarque :** les résultats de démonstration du Data Drift reposent sur des profils de test. Ils ne constituent pas une preuve de dérive réelle dans la clientèle de l'entreprise.

## 5. Structure du projet

```text
Telecom_Churn_MLOps/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── data/
│   └── Telco-Customer-Churn.csv
│
├── logs/
│   ├── api.log
│   ├── predictions.csv
│   └── prediction_inputs.csv
│
├── models/
│   └── random_forest_churn.joblib
│
├── reports/
│   └── data_drift_report.csv
│
├── src/
│   ├── data.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   ├── api.py
│   ├── monitoring.py
│   └── drift.py
│
├── tests/
│   ├── test_api.py
│   ├── test_data.py
│   └── test_features.py
│
├── dashboard.py
├── Dockerfile
├── requirements.txt
└── README.md
```

## 6. Installation et configuration

### Prérequis
- Python
- Git
- Docker Desktop
- Un éditeur de code, par exemple Visual Studio Code

### Cloner le projet

```bash
git clone https://github.com/Rouddlygilot/Telecom_Churn_MLOps.git
cd Telecom_Churn_MLOps
```

### Créer un environnement virtuel

```bash
python -m venv env
```

### Activer l'environnement virtuel sous Windows

```bash
env\Scripts\activate
```

### Installer les dépendances

```bash
pip install -r requirements.txt
```

## 7. Exécution du projet

### 7.1 Lancer l'API

À la racine du projet :

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Accéder à l'API :

- Accueil : http://localhost:8000
- Documentation interactive : http://localhost:8000/docs

### 7.2 Lancer le tableau de bord

Dans un deuxième terminal, avec le même environnement virtuel activé :

```bash
streamlit run dashboard.py
```

Accéder au tableau de bord :

http://localhost:8501

### 7.3 Exécuter les tests automatisés

```bash
pytest -v
```

Le projet comporte 13 tests automatisés validant les principales fonctionnalités.

### 7.4 Exécuter l'analyse de Data Drift

```bash
python -m src.drift
```

Le rapport est généré dans :

```text
reports/data_drift_report.csv
```

### 7.5 Construire et lancer le conteneur Docker

Construire l'image :

```bash
docker build -t telecom-churn-api .
```

Lancer le conteneur :

```bash
docker run -d --name telecom-churn-container -p 8000:8000 -v "%cd%\logs:/app/logs" telecom-churn-api
```

Vérifier son fonctionnement :

```bash
docker ps
```

Consulter les journaux :

```bash
docker logs telecom-churn-container
```

## 8. Intégration continue avec GitHub Actions

Le projet utilise GitHub Actions pour automatiser l'exécution des tests lors des événements configurés dans le workflow.

Le fichier de configuration est :

```text
.github/workflows/ci.yml
```

Cette intégration continue permet de détecter plus rapidement les erreurs et de vérifier que les fonctionnalités principales restent opérationnelles.

## 9. Fichiers de surveillance

| Fichier | Rôle |
|---|---|
| `logs/predictions.csv` | Historique des prédictions |
| `logs/prediction_inputs.csv` | Données transmises au modèle |
| `logs/api.log` | Journal des requêtes et erreurs de l'API |
| `reports/data_drift_report.csv` | Résultats de l'analyse de Data Drift |

Les prédictions représentent les résultats estimés par le modèle. Elles ne doivent pas être confondues avec les désabonnements réellement observés. Le suivi des performances réelles nécessitera de disposer ultérieurement des résultats effectifs des clients.

## 10. Conclusion

Ce projet met en pratique les principales étapes de l'industrialisation d'un modèle de Machine Learning : développement, déploiement, automatisation des tests et surveillance.

L'utilisation de FastAPI, Docker, Streamlit, Pytest et GitHub Actions permet de construire une application de prédiction structurée et maintenable, tout en introduisant les principes essentiels d'une démarche MLOps.