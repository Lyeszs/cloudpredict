import os
import json
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
import mlflow

# ---- CONFIGURATION DU TP ----
# REMPLACE PAR L'URL DONNÉE PAR MINIKUBE (ex: http://192.168.49.2:30500)
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://192.168.49.2:30500") 
EXPERIMENT_NAME = "cloudpredict-xgboost" # Nouveau nom pour repartir à zéro
MODEL_NAME = "housing-price-predictor"
REFERENCE_DATA_PATH = os.environ.get("REFERENCE_DATA_PATH", "ml/reference_data.csv")
MODEL_OUTPUT_PATH = os.environ.get("MODEL_OUTPUT_PATH", "ml/models/xgb_model.joblib")

def load_data():
    housing = fetch_california_housing()
    return housing.data, housing.target, housing.feature_names

def engineer(X, names):
    df = {n: X[:, i] for i, n in enumerate(names)}
    extra = np.column_stack([
        df["AveRooms"] / (df["AveOccup"] + 1e-5),
        df["AveBedrms"] / (df["AveRooms"] + 1e-5),
        df["Population"] / (df["AveOccup"] + 1e-5),
        df["MedInc"] / (df["AveRooms"] + 1e-5),
        df["Latitude"] * df["Longitude"],
        np.sqrt((df["Latitude"] - 36.7)**2 + (df["Longitude"] - (-119.4))**2),
        np.log1p(df["Population"]),
        np.log1p(df["AveOccup"]),
        np.log1p(df["MedInc"]),
        df["MedInc"] ** 2,
        df["AveOccup"] ** 2
    ])
    return np.hstack([X, extra])

def add_clusters(X, kmeans_model):
    coords = X[:, 6:8]
    clusters = kmeans_model.predict(coords).reshape(-1, 1)
    dists = kmeans_model.transform(coords).min(axis=1).reshape(-1, 1)
    return np.hstack([X, clusters, dists])

def save_reference_data(X_raw, y, feature_names, path):
    """Sauvegarde les données d'origine pour la détection de drift (Étape 5)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(X_raw, columns=feature_names)
    df["target"] = y
    df.to_csv(path, index=False)
    print(f"[INFO] Donnees de reference sauvegardees dans {path}")

def main():
    print("=" * 60)
    print("CloudPredict - Entrainement du modele XGBoost + MLflow")
    print("=" * 60)
    
    # Configuration MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    with mlflow.start_run(run_name="training-xgb-run") as run:
        # [1/4] Chargement
        print("\n[1/4] Chargement du dataset...")
        X_raw, y, names = load_data()
        X_eng = engineer(X_raw, names)
        
        # [2/4] Split
        print("\n[2/4] Division train/test...")
        # On split aussi les données brutes pour sauvegarder la référence propre pour Evidently
        X_train_raw, _, y_train_raw, _ = train_test_split(X_raw, y, test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X_eng, y, test_size=0.2, random_state=42)
        
        # Sauvegarde pour le TP (Evidently a besoin des données brutes avec les vrais noms de colonnes)
        save_reference_data(X_train_raw, y_train_raw, names, REFERENCE_DATA_PATH)
        
        # [3/4] Entrainement
        print("\n[3/4] Entrainement du XGBoostRegressor...")
        km = KMeans(n_clusters=30, random_state=42, n_init=10)
        km.fit(X_train[:, 6:8])
        X_train_clust = add_clusters(X_train, km)
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_clust)
        
        best_params = {
            'n_estimators': 2684,
            'learning_rate': 0.023463258960422607,
            'max_depth': 6,
            'min_child_weight': 3,
            'subsample': 0.909388760000992,
            'colsample_bytree': 0.8442245426717814,
            'gamma': 0.0001307453591362648,
            'reg_alpha': 0.5844805204033486,
            'reg_lambda': 2.0006219005796497,
            'random_state': 42,
            'n_jobs': -1
        }
        
        mlflow.log_params(best_params) # Requis par le barème
        
        model = xgb.XGBRegressor(**best_params)
        model.fit(X_train_s, y_train, verbose=False)
        print(" - Entrainement termine !")
        
        # [4/4] Evaluation
        print("\n[4/4] Evaluation du modele...")
        X_test_clust = add_clusters(X_test, km)
        X_test_s = scaler.transform(X_test_clust)
        y_pred = model.predict(X_test_s)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f" - MAE : {mae:.4f}")
        print(f" - RMSE : {rmse:.4f}")
        print(f" - R2 : {r2:.4f}")
        
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        
        # --- SATISFAIRE LE BARÈME MLOPS ---
        
        # 1. Enregistrer le modèle dans le registry
        # On convertit un exemple de test en DataFrame pour la "signature" du modèle
        input_example = pd.DataFrame(X_test_s[:1]) 
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
            input_example=input_example
        )
        print(f"[INFO] Modele enregistre dans le Model Registry : {MODEL_NAME}")
        
        # 2. Logger les Feature Importances (Format attendu par le TP)
        importances = {f"feature_{i}": float(imp) for i, imp in enumerate(model.feature_importances_)}
        with open("ml/feature_importances.json", "w") as f:
            json.dump(importances, f, indent=2)
        mlflow.log_artifact("ml/feature_importances.json", artifact_path="analysis")
        mlflow.log_artifact(REFERENCE_DATA_PATH, artifact_path="data")

        # 3. Sauvegarde Locale des artefacts (incluant tes scalers/kmeans custom)
        models_dir = os.path.dirname(MODEL_OUTPUT_PATH)
        os.makedirs(models_dir, exist_ok=True)
        joblib.dump(model, MODEL_OUTPUT_PATH)
        joblib.dump(scaler, os.path.join(models_dir, "scaler.joblib"))
        joblib.dump(km, os.path.join(models_dir, "kmeans.joblib"))
        
        print(f"[INFO] MLflow Run ID : {run.info.run_id}")

    print("\n" + "=" * 60)
    print("Pipeline termine et donnees envoyees a MLflow !")
    print("=" * 60)

if __name__ == "__main__":
    main()