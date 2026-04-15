# ml/drift_detector.py
"""
Detection de drift de donnees avec Evidently.
Genere un rapport HTML et logge les metriques dans MLflow.
"""
import os
import sys
import json
import mlflow
import pandas as pd
import numpy as np
from datetime import datetime
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from sklearn.datasets import fetch_california_housing

# ---- Configuration ----
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://192.168.49.2:30500")
EXPERIMENT_NAME = "cloudpredict-xgboost" # On garde ton experience
REFERENCE_DATA_PATH = os.environ.get("REFERENCE_DATA_PATH", "ml/reference_data.csv")
CURRENT_DATA_PATH = os.environ.get("CURRENT_DATA_PATH", "ml/current_data.csv")
REPORTS_DIR = os.environ.get("REPORTS_DIR", "ml/reports")

def load_reference_data(path):
    if not os.path.exists(path):
        print(f"[ERREUR] Fichier de reference introuvable : {path}")
        sys.exit(1)
    df = pd.read_csv(path)
    print(f"[INFO] Donnees de reference chargees ({df.shape[0]} lignes)")
    return df

def generate_simulated_data(n_samples=500, drift_level=0.0):
    """Genere des donnees simulees avec un niveau de drift controlable."""
    housing = fetch_california_housing(as_frame=True)
    X = housing.data.copy()
    y = housing.target.copy()
    
    sample_idx = np.random.choice(len(X), size=n_samples, replace=False)
    df = X.iloc[sample_idx].copy()
    df["target"] = y.iloc[sample_idx].values
    
    if drift_level > 0:
        print(f"[INFO] Application de drift simule (niveau: {drift_level})")
        df["MedInc"] = df["MedInc"] * (1 + drift_level * 0.8)
        df["HouseAge"] = df["HouseAge"] + (drift_level * 15)
        df["AveRooms"] = df["AveRooms"] * (1 + drift_level * 0.5)
        df["Population"] = df["Population"] * (1 + drift_level * 0.6)
    return df

def detect_drift(reference_data, current_data):
    feature_cols = [col for col in reference_data.columns if col != "target"]
    ref = reference_data[feature_cols].copy()
    cur = current_data[feature_cols].copy()
    
    print("[INFO] Execution de la detection de drift Evidently...")
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)
    
    result = report.as_dict()
    
    # Recherche securisée pour parer aux changements de version d'Evidently
    dataset_drift_results = {}
    columns_drift_results = {}
    
    for metric in result["metrics"]:
        res = metric.get("result", {})
        if "dataset_drift" in res:
            dataset_drift_results = res
        if "drift_by_columns" in res:
            columns_drift_results = res
            
    feature_drift = {}
    if "drift_by_columns" in columns_drift_results:
        for col_name, col_data in columns_drift_results["drift_by_columns"].items():
            feature_drift[col_name] = {
                "drifted": col_data.get("drift_detected", False),
                "drift_score": round(col_data.get("drift_score", 0.0), 4)
            }
        
    summary = {
        "dataset_drift_detected": dataset_drift_results.get("dataset_drift", False),
        "drift_share": round(dataset_drift_results.get("drift_share", 0.0), 4),
        "n_drifted_columns": dataset_drift_results.get("number_of_drifted_columns", 0),
        "n_total_columns": dataset_drift_results.get("number_of_columns", 0),
        "feature_drift": feature_drift
    }
    return report, summary
def save_report(report, reports_dir):
    os.makedirs(reports_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(reports_dir, f"drift_report_{timestamp}.html")
    report.save_html(report_path)
    print(f"[INFO] Rapport HTML sauvegarde : {report_path}")
    return report_path

def log_to_mlflow(summary, report_path):
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        with mlflow.start_run(run_name="drift-detection"):
            mlflow.log_metric("drift_detected", 1.0 if summary["dataset_drift_detected"] else 0.0)
            mlflow.log_metric("drift_share", summary["drift_share"])
            mlflow.log_metric("n_drifted_columns", summary["n_drifted_columns"])
            mlflow.log_artifact(report_path, artifact_path="drift_reports")
            print("[INFO] Metriques de drift enregistrees dans MLflow")
    except Exception as e:
        print(f"[AVERTISSEMENT] Impossible de logger dans MLflow : {e}")

def main():
    drift_level = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    print("=" * 60 + "\n DETECTION DE DRIFT \n" + "=" * 60)
    
    reference_data = load_reference_data(REFERENCE_DATA_PATH)
    current_data = generate_simulated_data(n_samples=500, drift_level=drift_level)
    
    report, summary = detect_drift(reference_data, current_data)
    report_path = save_report(report, REPORTS_DIR)
    log_to_mlflow(summary, report_path)
    
    print("\n RESULTATS :")
    print(f" Drift detecte : {'OUI' if summary['dataset_drift_detected'] else 'NON'}")
    print(f" Part de drift : {summary['drift_share']:.2%}")
    print(f" Colonnes en drift : {summary['n_drifted_columns']}/{summary['n_total_columns']}")

if __name__ == "__main__":
    main()