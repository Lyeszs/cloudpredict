# ml/retrain.py
"""
Pipeline de reentrainement automatique adapté pour XGBoost.
1. Verifie si du drift est detecte
2. Si oui, refait l'Engineering, le KMeans, le Scaler et le XGBoost
3. Enregistre la nouvelle version dans MLflow
"""
import os
import sys
import mlflow
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.tracking import MlflowClient

# Import de tes fonctions persos
from ml.train import engineer, add_clusters
from ml.drift_detector import detect_drift, generate_simulated_data, load_reference_data, save_report
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# ---- Configuration ----
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://192.168.49.2:30500")
EXPERIMENT_NAME = "cloudpredict-xgboost"
MODEL_NAME = "housing-price-predictor"
REFERENCE_DATA_PATH = os.environ.get("REFERENCE_DATA_PATH", "ml/reference_data.csv")
REPORTS_DIR = os.environ.get("REPORTS_DIR", "ml/reports")

# Dossier local (fallback)
BASE_PATH = os.path.join(os.path.dirname(__file__), "models")
MODEL_OUTPUT_PATH = os.path.join(BASE_PATH, "xgb_model.joblib")
SCALER_OUTPUT_PATH = os.path.join(BASE_PATH, "scaler.joblib")
KMEANS_OUTPUT_PATH = os.path.join(BASE_PATH, "kmeans.joblib")

def check_drift():
    print("[ETAPE 1] Verification du drift...")
    ref_data = load_reference_data(REFERENCE_DATA_PATH)
    # On simule un drift fort pour forcer le reentrainement lors du test
    cur_data = generate_simulated_data(n_samples=500, drift_level=1.0) 
    report, summary = detect_drift(ref_data, cur_data)
    save_report(report, REPORTS_DIR)
    
    drift_detected = summary["dataset_drift_detected"]
    print(f"[INFO] Drift detecte : {'OUI' if drift_detected else 'NON'}")
    return drift_detected, summary

def get_current_production_metrics(client):
    try:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        prod_versions = [v for v in versions if v.current_stage == "Production"]
        if not prod_versions:
            return None
        latest_prod = max(prod_versions, key=lambda v: int(v.version))
        run = client.get_run(latest_prod.run_id)
        print(f"[INFO] Modele Production actuel : v{latest_prod.version} | R2: {run.data.metrics.get('r2', 0):.4f}")
        return run.data.metrics
    except Exception as e:
        print(f"[AVERTISSEMENT] Impossible de recuperer les metriques actuelles : {e}")
        return None

def retrain_model():
    print("[ETAPE 2] Reentrainement du pipeline complet (XGBoost)...")
    housing = fetch_california_housing()
    X_raw, y, names = housing.data, housing.target, housing.feature_names
    
    # Nouveau split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42)
    
    # Sauvegarde des nouvelles donnees de reference
    df_ref = pd.DataFrame(X_train_raw, columns=names)
    df_ref["target"] = y_train
    df_ref.to_csv(REFERENCE_DATA_PATH, index=False)
    
    # 1. Feature Engineering
    X_train_eng = engineer(X_train_raw, names)
    X_test_eng = engineer(X_test_raw, names)
    
    # 2. KMeans
    km = KMeans(n_clusters=30, random_state=42, n_init=10)
    km.fit(X_train_eng[:, 6:8])
    X_train_clust = add_clusters(X_train_eng, km)
    X_test_clust = add_clusters(X_test_eng, km)
    
    # 3. Scaler
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_clust)
    X_test_s = scaler.transform(X_test_clust)
    
    # 4. Entrainement XGBoost
    params = {
        'n_estimators': 1500, 'learning_rate': 0.05, 'max_depth': 6, 
        'random_state': 42, 'n_jobs': -1
    }
    model = xgb.XGBRegressor(**params)
    model.fit(X_train_s, y_train, verbose=False)
    
    # Evaluation
    y_pred = model.predict(X_test_s)
    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred)
    }
    print(f"[INFO] Nouvelles metriques -> MAE: {metrics['mae']:.4f}, R2: {metrics['r2']:.4f}")
    
    # Sauvegarde locale
    os.makedirs(BASE_PATH, exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT_PATH)
    joblib.dump(scaler, SCALER_OUTPUT_PATH)
    joblib.dump(km, KMEANS_OUTPUT_PATH)
    
    return model, metrics, X_test_s

def register_and_promote(model, metrics, X_test_s, client):
    print("[ETAPE 3] Enregistrement dans MLflow et Promotion...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    with mlflow.start_run(run_name=f"retrain-{datetime.now().strftime('%Y%m%d-%H%M')}") as run:
        mlflow.log_metrics(metrics)
        mlflow.log_param("trigger", "drift_detected")
        
        input_example = pd.DataFrame(X_test_s[:1])
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
            input_example=input_example
        )
        
    # Promotion
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    latest = max(versions, key=lambda v: int(v.version))
    
    client.transition_model_version_stage(
        name=MODEL_NAME, version=latest.version, stage="Production", archive_existing_versions=True
    )
    print(f"[INFO] Nouveau modele (v{latest.version}) promu en Production !")
    print("[INFO] L'ancienne version a ete archivee.")

def main():
    print("="*60 + "\n PIPELINE DE REENTRAINEMENT \n" + "="*60)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    
    force = "--force" in sys.argv
    drift_detected, summary = check_drift()
    
    if not drift_detected and not force:
        print("\n[RESULTAT] Aucun drift detecte. Pas de reentrainement.")
        return

    current_metrics = get_current_production_metrics(client)
    model, new_metrics, X_test_s = retrain_model()
    
    # Tolerance : on promeut si le modele n'est pas catastrophiquement pire
    improvement = new_metrics["r2"] - (current_metrics.get("r2", 0) if current_metrics else 0)
    
    if improvement >= -0.05: 
        print(f"[INFO] Performance validee (diff R2: {improvement:+.4f}) -> PROMOTION")
        register_and_promote(model, new_metrics, X_test_s, client)
    else:
        print(f"[INFO] Regression forte ({improvement:+.4f}) -> PAS DE PROMOTION")

if __name__ == "__main__":
    main()