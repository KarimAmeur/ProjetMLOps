import streamlit as st
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io

# Configuration de la page
st.set_page_config(
    page_title="Application de Prédiction de Défaut",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fonction pour charger les expériences MLflow
def load_mlflow_experiments(experiment_type):
    """Charger les données MLflow pour un type d'expérience spécifique"""
    # Dans un cas réel, utilisez mlflow.search_runs() avec les bons filtres
    # Exemple: runs = mlflow.search_runs(filter_string=f"tags.experiment_type = '{experiment_type}'")
    
    # Pour la démonstration, on crée des données factices
    if experiment_type == "logistic_regression":
        return {
            "params": {
                "solver": "liblinear",
                "max_iter": 100
            },
            "metrics": {
                "accuracy": 0.85,
                "precision": 0.84,
                "recall": 0.85,
                "f1_score": 0.84
            },
            "artifacts": {
                "confusion_matrix": np.array([[850, 150], [200, 800]]),
                "model_path": "runs:/logistic_regression_model/model"
            }
        }
    elif experiment_type == "random_forest":
        return {
            "params": {
                "n_estimators": 100,
                "random_state": 42
            },
            "metrics": {
                "accuracy": 0.89,
                "precision": 0.88,
                "recall": 0.89,
                "f1_score": 0.88
            },
            "artifacts": {
                "confusion_matrix": np.array([[900, 100], [150, 850]]),
                "model_path": "runs:/random_forest_model/model"
            }
        }

# Fonction pour charger un modèle MLflow
def load_model(model_path):
    """
    Charge un modèle MLflow à partir de son chemin
    
    Dans un environnement réel, utilisez:
    return mlflow.sklearn.load_model(model_path)
    """
    # Pour la démo, on retourne un modèle factice
    if "logistic_regression" in model_path:
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression()
    else:
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier()

# Fonction pour faire une prédiction
def predict(model, features):
    """Effectue une prédiction avec le modèle choisi"""
    # Conversion des valeurs en float
    features_float = [float(val) for val in features]
    # Création d'un DataFrame avec les bonnes colonnes
    feature_df = pd.DataFrame([features_float], columns=[
        'credit_lines_outstanding', 'loan_amt_outstanding', 
        'total_debt_outstanding', 'income', 
        'years_employed', 'fico_score'
    ])
    
    # Dans un environnement réel, utilisez:
    # return model.predict(feature_df)[0], model.predict_proba(feature_df)[0][1]
    
    # Pour la démo, on simule une prédiction basée sur le score FICO
    if features_float[5] > 700:  # FICO score
        return 0, 0.15  # Non défaut, probabilité de défaut de 15%
    else:
        return 1, 0.75  # Défaut, probabilité de défaut de 75%

# Fonction pour créer un plot de matrice de confusion
def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    plt.xlabel('Prédiction')
    plt.ylabel('Valeur réelle')
    plt.title('Matrice de Confusion')
    ax.set_xticklabels(['Non défaut', 'Défaut'])
    ax.set_yticklabels(['Non défaut', 'Défaut'])
    
    return fig

# Fonction pour créer un graphique de métriques
def plot_metrics(metrics):
    metrics_df = pd.DataFrame(list(metrics.items()), columns=['Métrique', 'Valeur'])
    
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x='Métrique', y='Valeur', data=metrics_df, ax=ax)
    plt.ylim(0, 1)
    plt.title('Métriques de Performance')
    
    return fig

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Sélectionnez une page", ["Régression Logistique", "Random Forest", "Prédiction"])

# Page 1: Régression Logistique
if page == "Régression Logistique":
    st.title("🔍 Visualisation des résultats de la Régression Logistique")
    
    # Chargement des données MLflow
    lr_data = load_mlflow_experiments("logistic_regression")
    
    # Affichage des hyperparamètres
    st.header("Hyperparamètres")
    st.write(pd.DataFrame(lr_data["params"].items(), columns=["Paramètre", "Valeur"]))
    
    # Affichage des métriques
    st.header("Métriques de performance")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(pd.DataFrame(lr_data["metrics"].items(), columns=["Métrique", "Valeur"]))
    
    with col2:
        metrics_fig = plot_metrics(lr_data["metrics"])
        st.pyplot(metrics_fig)
    
    # Matrice de confusion
    st.header("Matrice de confusion")
    cm_fig = plot_confusion_matrix(lr_data["artifacts"]["confusion_matrix"])
    st.pyplot(cm_fig)
    
    # Informations supplémentaires
    st.header("Informations supplémentaires")
    st.write("""
    Ce modèle de régression logistique a été entraîné pour prédire la probabilité de défaut de paiement 
    d'un client en fonction de diverses caractéristiques financières. La régression logistique est 
    particulièrement adaptée pour les problèmes de classification binaire comme celui-ci.
    """)

# Page 2: Random Forest
elif page == "Random Forest":
    st.title("🌲 Visualisation des résultats du Random Forest")
    
    # Chargement des données MLflow
    rf_data = load_mlflow_experiments("random_forest")
    
    # Affichage des hyperparamètres
    st.header("Hyperparamètres")
    st.write(pd.DataFrame(rf_data["params"].items(), columns=["Paramètre", "Valeur"]))
    
    # Affichage des métriques
    st.header("Métriques de performance")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(pd.DataFrame(rf_data["metrics"].items(), columns=["Métrique", "Valeur"]))
    
    with col2:
        metrics_fig = plot_metrics(rf_data["metrics"])
        st.pyplot(metrics_fig)
    
    # Matrice de confusion
    st.header("Matrice de confusion")
    cm_fig = plot_confusion_matrix(rf_data["artifacts"]["confusion_matrix"])
    st.pyplot(cm_fig)
    
    # Informations supplémentaires
    st.header("Informations supplémentaires")
    st.write("""
    Le modèle Random Forest est un ensemble de 100 arbres de décision qui ont été entraînés pour 
    prédire la probabilité de défaut de paiement. Les modèles d'ensemble comme Random Forest sont 
    souvent plus robustes et peuvent capturer des relations non linéaires complexes dans les données.
    """)

# Page 3: Prédiction
else:
    st.title("🔮 Prédiction de défaut de paiement")
    
    # Sélection du modèle
    model_type = st.selectbox(
        "Sélectionnez le modèle à utiliser",
        ["Régression Logistique", "Random Forest"]
    )
    
    # Chargement du modèle sélectionné
    if model_type == "Régression Logistique":
        model_path = load_mlflow_experiments("logistic_regression")["artifacts"]["model_path"]
    else:
        model_path = load_mlflow_experiments("random_forest")["artifacts"]["model_path"]
    
    model = load_model(model_path)
    
    # Formulaire de saisie des caractéristiques
    st.header("Saisie des caractéristiques du client")
    
    col1, col2 = st.columns(2)
    
    with col1:
        credit_lines = st.number_input("Nombre de lignes de crédit", min_value=0, value=3)
        loan_amt = st.number_input("Montant du prêt en cours ($)", min_value=0, value=15000)
        total_debt = st.number_input("Dette totale ($)", min_value=0, value=35000)
    
    with col2:
        income = st.number_input("Revenu annuel ($)", min_value=0, value=60000)
        years_employed = st.number_input("Années d'emploi", min_value=0.0, value=5.0)
        fico_score = st.number_input("Score FICO", min_value=300, max_value=850, value=680)
    
    # Bouton de prédiction
    if st.button("Faire une prédiction"):
        features = [credit_lines, loan_amt, total_debt, income, years_employed, fico_score]
        prediction, proba = predict(model, features)
        
        st.header("Résultat de la prédiction")
        
        # Affichage du résultat avec un indicateur visuel
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 0:
                st.success("✅ Non défaut de paiement prédit")
            else:
                st.error("❌ Défaut de paiement prédit")
            
            st.write(f"Probabilité de défaut: {proba:.2%}")
        
        with col2:
            # Graphique de probabilité
            fig, ax = plt.subplots(figsize=(5, 3))
            
            # Création d'une jauge de risque
            categories = ['Faible', 'Moyen', 'Élevé']
            colors = ['green', 'orange', 'red']
            bounds = [0, 0.3, 0.7, 1.0]
            
            plt.barh(categories, [0.3, 0.4, 0.3], color=colors, alpha=0.3)
            plt.barh([''], [proba], left=0, color='blue', height=0.3)
            
            # Ajout d'un marqueur pour la probabilité
            plt.axvline(x=proba, color='blue', linestyle='-', alpha=0.7)
            plt.text(proba, 0, f"{proba:.2%}", color='blue', fontweight='bold')
            
            plt.xlim(0, 1)
            plt.title('Niveau de risque')
            plt.xticks([0, 0.25, 0.5, 0.75, 1], ['0%', '25%', '50%', '75%', '100%'])
            
            st.pyplot(fig)
        
        # Informations explicatives
        st.header("Facteurs influençant la prédiction")
        
        if model_type == "Random Forest" and fico_score < 700:
            st.write("""
            Le score FICO est l'un des facteurs les plus importants dans cette prédiction. 
            Un score inférieur à 700 est généralement associé à un risque de défaut plus élevé.
            """)
        elif total_debt > income * 0.5:
            st.write("""
            Le ratio dette/revenu est élevé (plus de 50% du revenu annuel), 
            ce qui est généralement un indicateur de risque de défaut.
            """)
        else:
            st.write("""
            Plusieurs facteurs ont contribué à cette prédiction, notamment la combinaison du score FICO, 
            du montant de la dette et des années d'emploi.
            """)

# Footer
st.sidebar.markdown("---")
st.sidebar.info(
    "Cette application démontre l'utilisation de MLflow pour le suivi et le déploiement "
    "de modèles de prédiction de défaut de paiement."
)