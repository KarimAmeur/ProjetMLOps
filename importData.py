import pandas as pd
import pyarrow as pa


def load_data(file_path):
    # Chemin du fichier
    file_path = file_path

    # Lecture du fichier dans un DataFrame
    df = pd.read_csv(file_path, delimiter=',')  # Utilisez '\t' si les colonnes sont séparées par des tabulations
    df = df.drop(columns=['customer_id'])
    
    try:
        table = pa.Table.from_pandas(df)
    except Exception as e:
        print(f"Erreur lors de la conversion : {e}")


    # Affichage des premières lignes du DataFrame pour vérifier
    print(df.head())

    return df
file_path = 'Loan_Data.csv'
    
# Charger les données dans un DataFrame
df = load_data(file_path)