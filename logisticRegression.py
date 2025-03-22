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
    X = df.drop(columns=["default"])
    y = df["default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model_lr = LogisticRegression()

    with mlflow.start_run():
        model_lr.fit(X_train, y_train)

        y_pred = model_lr.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        mlflow.log_param("solver", "liblinear")
        mlflow.log_param("max_iter", 100)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", report["accuracy"])
        mlflow.log_metric("recall", report["weighted avg"]["recall"])
        mlflow.log_metric("f1_score", report["weighted avg"]["f1-score"])

        cm = confusion_matrix(y_test, y_pred)
        mlflow.log_text(str(cm), "confusion_matrix.txt")

        mlflow.sklearn.log_model(model_lr, "logistic_regression_model")

        mlflow.set_tag("experiment_type", "logistic_regression")

        print("Accuracy:", accuracy)
        print("\nMatrice de confusion:")
        print(cm)
        print("\nRapport de classification:")
        print(report)
    return model_lr


def random_forest(df):
    """
    Effectue un Random Forest avec 100 arbres sur le DataFrame pour prédire la colonne 'default'
    et enregistre les résultats dans MLflow.

    Args:
        df (pd.DataFrame): DataFrame contenant les données.
    """
    X = df.drop(columns=["default"])
    y = df["default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model_rf = RandomForestClassifier(n_estimators=100, random_state=42)

    with mlflow.start_run():
        model_rf.fit(X_train, y_train)

        y_pred = model_rf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", report["accuracy"])
        mlflow.log_metric("recall", report["weighted avg"]["recall"])
        mlflow.log_metric("f1_score", report["weighted avg"]["f1-score"])

        mlflow.log_text(str(cm), "confusion_matrix.txt")
        mlflow.log_text(str(report), "classification_report.txt")

        mlflow.sklearn.log_model(model_rf, "random_forest_model")

        mlflow.set_tag("experiment_type", "random_forest")

        print("Accuracy:", accuracy)
        print("\nMatrice de confusion:")
        print(cm)
        print("\nRapport de classification:")
        print(report)
    return model_rf
