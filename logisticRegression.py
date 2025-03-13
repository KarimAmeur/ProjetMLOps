import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier


def logistic_regression(df):
    """
    Effectue une régression logistique sur le DataFrame pour prédire la colonne 'default' et 
    enregistre les résultats dans MLflow.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données.
    """
    # Séparation des features (X) et de la target (y)
    X = df.drop(columns=['default'])
    y = df['default']

    # Séparation en ensembles d'entraînement et de test (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialisation du modèle
    model = LogisticRegression()

    with mlflow.start_run():
        # Entraînement du modèle
        model.fit(X_train, y_train)

        # Prédictions
        y_pred = model.predict(X_test)

        # Calcul des métriques
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Enregistrer les hyperparamètres
        mlflow.log_param("solver", "liblinear")
        mlflow.log_param("max_iter", 100)
        
        # Enregistrement des métriques
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", report['accuracy'])
        mlflow.log_metric("recall", report['weighted avg']['recall'])
        mlflow.log_metric("f1_score", report['weighted avg']['f1-score'])

        # Enregistrement de la matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        mlflow.log_text(str(cm), "confusion_matrix.txt")

        # Enregistrement du modèle
        mlflow.sklearn.log_model(model, "logistic_regression_model")
        
        # Enregistrement d'un tag pour l'expérience
        mlflow.set_tag("experiment_type", "logistic_regression")
        
        # Affichage des résultats
        print("Accuracy:", accuracy)
        print("\nMatrice de confusion:")
        print(cm)
        print("\nRapport de classification:")
        print(report)


def random_forest(df):
    """
    Effectue un Random Forest avec 100 arbres sur le DataFrame pour prédire la colonne 'default'
    et enregistre les résultats dans MLflow.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données.
    """
    # Séparation des features (X) et de la target (y)
    X = df.drop(columns=['default'])  # Toutes les colonnes sauf 'default'
    y = df['default']  # La colonne 'default' est la variable cible

    # Séparation en ensembles d'entraînement et de test (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialisation du modèle Random Forest avec 100 arbres
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    with mlflow.start_run():
        # Entraînement du modèle sur l'ensemble d'entraînement
        model.fit(X_train, y_train)

        # Prédiction sur l'ensemble de test
        y_pred = model.predict(X_test)

        # Calcul des métriques
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        # Enregistrement des hyperparamètres
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)
        
        # Enregistrement des métriques
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", report['accuracy'])
        mlflow.log_metric("recall", report['weighted avg']['recall'])
        mlflow.log_metric("f1_score", report['weighted avg']['f1-score'])

        # Enregistrement de la matrice de confusion
        mlflow.log_text(str(cm), "confusion_matrix.txt")

        # Enregistrement du rapport de classification
        mlflow.log_text(str(report), "classification_report.txt")

        # Enregistrement du modèle
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        # Enregistrement d'un tag pour l'expérience
        mlflow.set_tag("experiment_type", "random_forest")
        
        # Affichage des résultats
        print("Accuracy:", accuracy)
        print("\nMatrice de confusion:")
        print(cm)
        print("\nRapport de classification:")
        print(report)
