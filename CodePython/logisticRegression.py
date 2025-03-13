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
    X_lr = df.drop(columns=['default'])
    y_lr = df['default']

    # Séparation en ensembles d'entraînement et de test (80% train, 20% test)
    X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X_lr, y_lr, test_size=0.2, random_state=42)

    # Initialisation du modèle de régression logistique
    model_lr = LogisticRegression()

    with mlflow.start_run():
        # Entraînement du modèle
        model_lr.fit(X_train_lr, y_train_lr)

        # Prédictions
        y_pred_lr = model_lr.predict(X_test_lr)

        # Calcul des métriques
        accuracy_lr = accuracy_score(y_test_lr, y_pred_lr)
        report_lr = classification_report(y_test_lr, y_pred_lr, output_dict=True)
        
        # Enregistrer les hyperparamètres
        mlflow.log_param("solver", "liblinear")
        mlflow.log_param("max_iter", 100)
        
        # Enregistrement des métriques
        mlflow.log_metric("accuracy_lr", accuracy_lr)
        mlflow.log_metric("precision_lr", report_lr['accuracy'])
        mlflow.log_metric("recall_lr", report_lr['weighted avg']['recall'])
        mlflow.log_metric("f1_score_lr", report_lr['weighted avg']['f1-score'])

        # Enregistrement de la matrice de confusion
        cm_lr = confusion_matrix(y_test_lr, y_pred_lr)
        mlflow.log_text(str(cm_lr), "confusion_matrix_lr.txt")

        # Enregistrement du modèle
        mlflow.sklearn.log_model(model_lr, "logistic_regression_model")
        
        # Enregistrement d'un tag pour l'expérience
        mlflow.set_tag("experiment_type_lr", "logistic_regression")
        
        # Affichage des résultats
        print("Accuracy (Logistic Regression):", accuracy_lr)
        print("\nMatrice de confusion (Logistic Regression):")
        print(cm_lr)
        print("\nRapport de classification (Logistic Regression):")
        print(report_lr)


def random_forest(df):
    """
    Effectue un Random Forest avec 100 arbres sur le DataFrame pour prédire la colonne 'default'
    et enregistre les résultats dans MLflow.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données.
    """
    # Séparation des features (X) et de la target (y)
    X_rf = df.drop(columns=['default'])  # Toutes les colonnes sauf 'default'
    y_rf = df['default']  # La colonne 'default' est la variable cible

    # Séparation en ensembles d'entraînement et de test (80% train, 20% test)
    X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)

    # Initialisation du modèle Random Forest avec 100 arbres
    model_rf = RandomForestClassifier(n_estimators=100, random_state=42)

    with mlflow.start_run():
        # Entraînement du modèle sur l'ensemble d'entraînement
        model_rf.fit(X_train_rf, y_train_rf)

        # Prédiction sur l'ensemble de test
        y_pred_rf = model_rf.predict(X_test_rf)

        # Calcul des métriques
        accuracy_rf = accuracy_score(y_test_rf, y_pred_rf)
        report_rf = classification_report(y_test_rf, y_pred_rf, output_dict=True)
        cm_rf = confusion_matrix(y_test_rf, y_pred_rf)

        # Enregistrement des hyperparamètres
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)
        
        # Enregistrement des métriques
        mlflow.log_metric("accuracy_rf", accuracy_rf)
        mlflow.log_metric("precision_rf", report_rf['accuracy'])
        mlflow.log_metric("recall_rf", report_rf['weighted avg']['recall'])
        mlflow.log_metric("f1_score_rf", report_rf['weighted avg']['f1-score'])

        # Enregistrement de la matrice de confusion
        mlflow.log_text(str(cm_rf), "confusion_matrix_rf.txt")

        # Enregistrement du rapport de classification
        mlflow.log_text(str(report_rf), "classification_report_rf.txt")

        # Enregistrement du modèle
        mlflow.sklearn.log_model(model_rf, "random_forest_model")
        
        # Enregistrement d'un tag pour l'expérience
        mlflow.set_tag("experiment_type_rf", "random_forest")
        
        # Affichage des résultats
        print("Accuracy (Random Forest):", accuracy_rf)
        print("\nMatrice de confusion (Random Forest):")
        print(cm_rf)
        print("\nRapport de classification (Random Forest):")
        print(report_rf)
