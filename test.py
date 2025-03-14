import importData as id
import logisticRegression as lr
import pandas as pd


# Chemin du fichier de données
file_path = 'Loan_Data.csv'  # Assurez-vous que le fichier se trouve dans le répertoire de travail
print("test3!!!!!!!ç")
    
 # Charger les données dans un DataFrame
df = id.load_data(file_path)
print("commit3")
# Exécuter les modèles
print("Exécution de la régression logistique...")
model=lr.logistic_regression(df)

# Nouvelle donnée pour la prédiction
new_data = pd.DataFrame([{
    'credit_lines_outstanding': 5,
    'loan_amt_outstanding': 60000,
    'total_debt_outstanding': 100000,
    'income': 90000,
    'years_employed': 8,
    'fico_score': 680
}])

# Prédiction avec le modèle entraîné
prediction = model.predict(new_data)

# Vérification du résultat attendu
assert prediction[0] == 1, "Incorrect prediction"
print("Test réussi : la prédiction est correcte.")
