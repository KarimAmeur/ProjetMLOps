import importData as id
import logisticRegression as lr
import pandas as pd
import unittest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")


# Chemin du fichier de données
file_path = "Loan_Data.csv"  # Assurez-vous que le fichier se trouve dans le répertoire de travail
print("test3!!!!!!!ç")

# Charger les données dans un DataFrame
df = id.load_data(file_path)
print("commit3")
# Exécuter les modèles
print("Exécution de la régression logistique...")
model = lr.logistic_regression(df)

# Nouvelle donnée pour la prédiction
new_data = pd.DataFrame(
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

# Prédiction avec le modèle entraîné
prediction = model.predict(new_data)

# Vérification du résultat attendu
assert prediction[0] == 1, "Incorrect prediction"
print("Test réussi : la prédiction est correcte.")

import importData as id
from logisticRegression import (
    random_forest,
)  # Importe la fonction random_forest depuis LogisticRegression.py
import pandas as pd
import unittest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Chemin du fichier de données
file_path = "Loan_Data.csv"
print("test3!!!!!!!ç")

# Charger les données dans un DataFrame
df = id.load_data(file_path)
print("commit3")

# Exécuter le modèle Random Forest
print("Exécution du modèle Random Forest...")
model_rf = random_forest(df)  # Appel de la fonction random_forest

# Nouvelle donnée pour la prédiction
new_data = pd.DataFrame(
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

# Prédiction avec le modèle entraîné
prediction_rf = model_rf.predict(new_data)

# Vérification du résultat attendu
assert prediction_rf[0] == 1, "Incorrect prediction"
print("Test réussi : la prédiction est correcte.")
