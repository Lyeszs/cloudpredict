import os
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# On doit réimporter les fonctions de transformation pour traiter le test set
from ml.train import engineer, add_clusters

def load_artifacts(models_dir):
    """Charge le modèle, le scaler et le kmeans."""
    model_path = os.path.join(models_dir, "xgb_model.joblib")
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    kmeans_path = os.path.join(models_dir, "kmeans.joblib")
    
    if not all(map(os.path.exists, [model_path, scaler_path, kmeans_path])):
        raise FileNotFoundError("Il manque un ou plusieurs artefacts dans le dossier models/.")
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    km = joblib.load(kmeans_path)
    
    print(f"Modèle, Scaler et KMeans chargés depuis : {models_dir}")
    return model, scaler, km

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred),
        "mean_prediction": np.mean(y_pred),
        "std_prediction": np.std(y_pred),
        "min_prediction": np.min(y_pred),
        "max_prediction": np.max(y_pred),
    }
    return metrics, y_pred

def main():
    print("=" * 60)
    print("CloudPredict - Evaluation du modèle Optimisé")
    print("=" * 60)
    
    print("\n[1/3] Chargement du dataset original et extraction du Test Set...")
    housing = fetch_california_housing()
    X_raw, y = housing.data, housing.target
    
    # On applique d'abord l'engineering initial
    X_eng = engineer(X_raw, housing.feature_names)
    
    # On refait exactement le même split pour retomber sur le bon Test Set
    _, X_test, _, y_test = train_test_split(X_eng, y, test_size=0.2, random_state=42)
    
    print("\n[2/3] Chargement des artefacts et application du Pipeline...")
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    model, scaler, km = load_artifacts(models_dir)
    
    # Application des clusters géographiques
    X_test_clust = add_clusters(X_test, km)
    
    # Application du scaling
    X_test_s = scaler.transform(X_test_clust)
    
    print("\n[3/3] Evaluation sur le jeu de test (21 features préparées)...")
    metrics, y_pred = evaluate(model, X_test_s, y_test)
    
    print("\n--- Metriques de performance ---")
    print(f" MAE      : {metrics['mae']:.4f}")
    print(f" RMSE     : {metrics['rmse']:.4f}")
    print(f" R2 Score : {metrics['r2']:.4f}")
    
    print("\n--- Statistiques des predictions ---")
    print(f" Moyenne    : {metrics['mean_prediction']:.4f}")
    print(f" Ecart-type : {metrics['std_prediction']:.4f}")
    print(f" Min        : {metrics['min_prediction']:.4f}")
    print(f" Max        : {metrics['max_prediction']:.4f}")
    
    print("\n--- Comparaison valeurs reelles vs predictions (5 premiers) ---")
    for i in range(5):
        print(f" Reel: {y_test[i]:.4f} | Predit: {y_pred[i]:.4f} | Erreur: {abs(y_test[i] - y_pred[i]):.4f}")
        
    print("\n" + "=" * 60)
    print("Evaluation terminee !")
    print("=" * 60)

if __name__ == "__main__":
    main()