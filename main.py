import importData as id
import logisticRegression as lr
import pandas as pd
import subprocess
import os

if __name__ == "__main__":
    # Chemin du fichier de données
    file_path = "Loan_Data.csv"

    # Charger les données dans un DataFrame
    df = id.load_data(file_path)
    print("commit3")
    # Exécuter les modèles
    print("Exécution de la régression logistique...")
    lr.logistic_regression(df)

    print("Exécution du Random Forest...")
    lr.random_forest(df)

    print("Lancement de l'application Streamlit...")
    # Lancer l'application Streamlit
    streamlit_path = os.path.join(os.path.dirname(__file__), "app.py")
    subprocess.run(["streamlit", "run", streamlit_path])
