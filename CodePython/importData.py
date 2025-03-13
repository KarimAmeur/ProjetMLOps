import pandas as pd
def load_data(file_path):
    # Chemin du fichier
    file_path = file_path

    # Lecture du fichier dans un DataFrame
    df = pd.read_csv(file_path, delimiter=',')  # Utilisez '\t' si les colonnes sont séparées par des tabulations
    X = df.drop(columns=['customer_id'])
    # Affichage des premières lignes du DataFrame pour vérifier
    print(df.head())

    return df