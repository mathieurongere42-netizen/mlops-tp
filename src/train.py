# src/train.py (version simplifiée et robuste)
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
import os
import joblib

# Configuration robuste de MLflow
def setup_mlflow():
    """Configure MLflow pour utiliser le dossier local"""
    mlflow.set_tracking_uri("file:///" + os.path.abspath("mlruns"))
    mlflow.set_experiment("Analyse de Sentiments Twitter")
    print(f"📊 MLflow configuré : {mlflow.get_tracking_uri()}")

def train_model_simple(model_name, pipeline, X_train, y_train):
    """Version simplifiée sans sauvegarde locale problématique"""
    with mlflow.start_run(run_name=model_name):
        print(f"🚀 Entraînement de {model_name}...")
        
        # Logger les paramètres principaux
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("tfidf_max_features", 5000)
        mlflow.log_param("tfidf_ngram_range", "(1, 2)")
        
        # Entraînement
        pipeline.fit(X_train, y_train)
        
        # Calcul métrique
        train_accuracy = pipeline.score(X_train, y_train)
        mlflow.log_metric("train_accuracy", train_accuracy)
        
        # Logger le modèle (seulement dans MLflow)
        mlflow.sklearn.log_model(pipeline, "model")
        
        print(f"✅ {model_name} terminé - Accuracy: {train_accuracy:.4f}")

if __name__ == "__main__":
    # Configuration MLflow
    setup_mlflow()
    
    # Charger les données
    train_path = os.path.join('data', 'train.csv')
    if not os.path.exists(train_path):
        print("❌ Fichier train.csv non trouvé. Exécutez d'abord preprocess.py")
        exit(1)
    
    print("📥 Chargement des données...")
    train_df = pd.read_csv(train_path)
    X_train = train_df['text'].astype(str)
    y_train = train_df['sentiment']
    
    print(f"📊 Données: {len(X_train)} échantillons")
    
    # Pipelines simples
    lr_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    nb_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('clf', MultinomialNB())
    ])

    # Entraînement
    print("\n" + "="*50)
    train_model_simple('LogisticRegression', lr_pipeline, X_train, y_train)
    train_model_simple('NaiveBayes', nb_pipeline, X_train, y_train)
    print("="*50)
    print("🎉 ENTRAÎNEMENT TERMINÉ !")
    
    # Vérification
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    exp = mlflow.get_experiment_by_name("Analyse de Sentiments Twitter")
    runs = client.search_runs(exp.experiment_id)
    print(f"📈 {len(runs)} run(s) dans MLflow")