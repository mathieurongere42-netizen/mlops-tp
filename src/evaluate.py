# src/evaluate.py
import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score, f1_score
import mlflow
import mlflow.sklearn
import os

def evaluate_model(model_name, pipeline, X_test, y_test):
    """Évalue un modèle et log les métriques avec MLflow"""
    with mlflow.start_run(run_name=f"{model_name}_evaluation", nested=True):
        # Prédictions
        predictions = pipeline.predict(X_test)
        
        # Calcul des métriques
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='weighted')
        
        # Logger les métriques
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        
        # Rapport de classification détaillé
        report = classification_report(y_test, predictions, target_names=['Négatif', 'Positif'], output_dict=True)
        
        # Logger les métriques détaillées
        mlflow.log_metric("precision_0", report['Négatif']['precision'])
        mlflow.log_metric("recall_0", report['Négatif']['recall'])
        mlflow.log_metric("f1_0", report['Négatif']['f1-score'])
        
        mlflow.log_metric("precision_1", report['Positif']['precision'])
        mlflow.log_metric("recall_1", report['Positif']['recall'])
        mlflow.log_metric("f1_1", report['Positif']['f1-score'])
        
        print(f"\n--- Rapport de Classification ({model_name}) ---")
        print(classification_report(y_test, predictions, target_names=['Négatif', 'Positif']))
        
        return report

if __name__ == "__main__":
    # Charger les données de test
    test_df = pd.read_csv(os.path.join('data', 'test.csv'))
    X_test = test_df['text'].astype(str)
    y_test = test_df['sentiment']
    
    # Définir l'expérience MLflow
    mlflow.set_experiment("Analyse de Sentiments Twitter")
    
    # Charger et évaluer les modèles
    results = {}
    
    # Évaluer le modèle de Régression Logistique
    print('Évaluation du modèle de Régression Logistique...')
    lr_pipeline = joblib.load(os.path.join('models', 'logistic_regression_pipeline.joblib'))
    lr_report = evaluate_model('LogisticRegression', lr_pipeline, X_test, y_test)
    results['LogisticRegression'] = lr_report

    # Évaluer le modèle Naive Bayes
    print('\nÉvaluation du modèle Naive Bayes...')
    nb_pipeline = joblib.load(os.path.join('models', 'naive_bayes_pipeline.joblib'))
    nb_report = evaluate_model('NaiveBayes', nb_pipeline, X_test, y_test)
    results['NaiveBayes'] = nb_report

    # Affichage d'un tableau comparatif
    print("\n--- Tableau Comparatif des Performances ---")
    comparison_data = {
        "Modèle": ["Régression Logistique", "Naive Bayes"],
        "Accuracy": [results['LogisticRegression']['accuracy'], results['NaiveBayes']['accuracy']],
        "F1-Score (Pondéré)": [
            results['LogisticRegression']['weighted avg']['f1-score'],
            results['NaiveBayes']['weighted avg']['f1-score']
        ],
        "Precision (Moyenne)": [
            results['LogisticRegression']['weighted avg']['precision'],
            results['NaiveBayes']['weighted avg']['precision']
        ],
        "Recall (Moyenne)": [
            results['LogisticRegression']['weighted avg']['recall'],
            results['NaiveBayes']['weighted avg']['recall']
        ]
    }
    
    results_df = pd.DataFrame(comparison_data)
    print(results_df.to_string(index=False))