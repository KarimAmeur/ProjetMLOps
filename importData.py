import pandas as pd
import pyarrow as pa

def load_data(file_path):
    # Chemin du fichier
    file_path = file_path

    # Lecture du fichier dans un DataFrame
    df = pd.read_csv(file_path, delimiter=',')  # Utilisez '\t' si les colonnes sont séparées par des tabulations
    df = df.drop(columns=['customer_id'], errors='ignore')  # Ignore si 'customer_id' n'existe pas

    # Vérification des colonnes présentes dans le DataFrame
    print("Colonnes du DataFrame :")
    print(df.columns)

    # Vérification des types des colonnes
    print("Types des colonnes avant conversion :")
    print(df.dtypes)

    # Optionnel : Vérification et conversion de la colonne "Valeur" si elle existe
    if 'Valeur' in df.columns:
        print("Types uniques dans la colonne 'Valeur' :")
        print(df['Valeur'].apply(type).value_counts())
        df["Valeur"] = pd.to_numeric(df["Valeur"], errors='coerce')  # Convertir en numérique
    else:
        print("La colonne 'Valeur' n'existe pas dans le DataFrame.")

    try:
        table = pa.Table.from_pandas(df)
        print("Conversion réussie en Arrow Table")
    except Exception as e:
        print(f"Erreur lors de la conversion : {e}")

    # Affichage des premières lignes du DataFrame pour vérifier
    print(df.head())

    return df

file_path = 'Loan_Data.csv'

# Charger les données dans un DataFrame
df = load_data(file_path)
