import pytest
import pandas as pd
import importData as ida
import logisticRegression as lr
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# Fonction de remplacement simple pour logistic_regression sans MLflow
def train_logistic_regression(df):
    """Version simplifiée de la fonction logistic_regression sans MLflow."""
    X = df.drop(columns=["default"])
    y = df["default"]
    model = LogisticRegression()
    model.fit(X, y)
    return model


# Fonction de remplacement simple pour random_forest sans MLflow
def train_random_forest(df):
    """Version simplifiée de la fonction random_forest sans MLflow."""
    X = df.drop(columns=["default"])
    y = df["default"]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model


# Fixture pour charger les données une seule fois pour tous les tests
@pytest.fixture(scope="module")
def loan_data():
    """Charge les données de prêt pour les tests."""
    file_path = "Loan_Data.csv"
    return ida.load_data(file_path)


# Fixture pour le modèle de régression logistique - SANS monkeypatch
@pytest.fixture(scope="module")
def logistic_model(loan_data):
    """Entraîne et retourne un modèle de régression logistique sans MLflow."""
    # Utiliser directement la fonction de remplacement
    return train_logistic_regression(loan_data)


# Fixture pour le modèle Random Forest
@pytest.fixture(scope="module")
def random_forest_model(loan_data):
    """Entraîne et retourne un modèle Random Forest sans MLflow."""
    return train_random_forest(loan_data)


# Données de test pour les prédictions
@pytest.fixture
def test_client_data():
    """Retourne un exemple de données client pour tester les prédictions."""
    return pd.DataFrame(
        [
            {
                "credit_lines_outstanding": 5,
                "loan_amt_outstanding": 60000,
                "total_debt_outstanding": 100000,
                "income": 90000,
                "years_employed": 8,
                "fico_score": 680,
            }
        ]
    )


# Données de test pour un client à faible risque
@pytest.fixture
def low_risk_client_data():
    """Retourne un exemple de données client à faible risque."""
    return pd.DataFrame(
        [
            {
                "credit_lines_outstanding": 2,
                "loan_amt_outstanding": 10000,
                "total_debt_outstanding": 15000,
                "income": 120000,
                "years_employed": 10,
                "fico_score": 780,
            }
        ]
    )


def test_data_loading():
    """Teste si le chargement des données fonctionne correctement."""
    file_path = "Loan_Data.csv"
    df = ida.load_data(file_path)
    assert df is not None
    assert not df.empty
    assert "default" in df.columns


def test_logistic_regression_prediction(logistic_model, test_client_data):
    """Teste les prédictions du modèle de régression logistique."""
    prediction = logistic_model.predict(test_client_data)
    assert prediction is not None
    # Vérification de base que la prédiction est valide
    assert prediction[0] in [0, 1]


def test_random_forest_prediction(random_forest_model, test_client_data):
    """Teste les prédictions du modèle Random Forest."""
    prediction = random_forest_model.predict(test_client_data)
    assert prediction is not None
    # Vérification de base que la prédiction est valide
    assert prediction[0] in [0, 1]


def test_model_prediction_type(logistic_model, test_client_data):
    """Teste le type de sortie des prédictions."""
    prediction = logistic_model.predict(test_client_data)
    assert isinstance(prediction, (list, pd.Series, tuple, type(pd.array([0]))))

    proba = logistic_model.predict_proba(test_client_data)
    assert (
        proba.shape[1] == 2
    )  # Vérifie que nous avons des probabilités pour les deux classes
    assert 0 <= proba[0][1] <= 1  # Vérifie que la probabilité est entre 0 et 1
