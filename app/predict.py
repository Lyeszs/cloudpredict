# app/predict.py
"""
API de prediction de prix immobiliers.
Adaptée pour XGBoost + Feature Engineering avec chargement MLflow.
"""
import os
import time
import logging
import joblib
import numpy as np
from flask import Flask, request, jsonify
import mlflow

# On importe tes fonctions de transformation depuis le dossier ml
from ml.train import engineer, add_clusters

# ---- Configuration ----
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.environ.get("MODEL_NAME", "housing-price-predictor")
MODEL_STAGE = os.environ.get("MODEL_STAGE", "Production")
PORT = int(os.environ.get("PORT", "8000"))

# Chemins locaux pour le fallback et les transformers
BASE_PATH = os.path.join(os.path.dirname(__file__), "..", "ml", "models")
LOCAL_MODEL_PATH = os.path.join(BASE_PATH, "xgb_model.joblib")
LOCAL_SCALER_PATH = os.path.join(BASE_PATH, "scaler.joblib")
LOCAL_KMEANS_PATH = os.path.join(BASE_PATH, "kmeans.joblib")

FEATURE_NAMES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms", 
    "Population", "AveOccup", "Latitude", "Longitude"
]

# ---- Logging ----
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Variables globales pour le pipeline
_model = None
_scaler = None
_km = None
model_source = "none"

def load_transformers_local():
    """Charge le scaler et le kmeans depuis le disque local."""
    global _scaler, _km
    try:
        _scaler = joblib.load(LOCAL_SCALER_PATH)
        _km = joblib.load(LOCAL_KMEANS_PATH)
        return True
    except Exception as e:
        logger.error(f"Erreur chargement transformers locaux : {e}")
        return False

def load_model_from_mlflow():
    """Charge le modele XGBoost depuis MLflow (stage Production) et les transformers en local."""
    global _model, model_source
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        logger.info(f"Chargement du modele depuis MLflow : {model_uri}")
        
        # Attention : mlflow.xgboost au lieu de mlflow.sklearn
        _model = mlflow.xgboost.load_model(model_uri)
        
        if load_transformers_local():
            model_source = f"mlflow:{MODEL_NAME}/{MODEL_STAGE}"
            logger.info(f"Pipeline charge avec succes depuis MLflow ({model_source})")
            return True
        return False
    except Exception as e:
        logger.warning(f"Impossible de charger depuis MLflow : {e}")
        return False

def load_model_from_local():
    """Charge l'intégralité du pipeline depuis les fichiers locaux (fallback)."""
    global _model, model_source
    try:
        if os.path.exists(LOCAL_MODEL_PATH):
            logger.info(f"Fallback : chargement du modele local {LOCAL_MODEL_PATH}")
            _model = joblib.load(LOCAL_MODEL_PATH)
            if load_transformers_local():
                model_source = f"local:{LOCAL_MODEL_PATH}"
                logger.info(f"Pipeline local charge avec succes ({model_source})")
                return True
        logger.error(f"Fichier modele local introuvable : {LOCAL_MODEL_PATH}")
        return False
    except Exception as e:
        logger.error(f"Erreur chargement modele local : {e}")
        return False

def ensure_model_loaded():
    """S'assure qu'un modele est charge (MLflow d'abord, puis local)."""
    if _model is not None and _scaler is not None and _km is not None:
        return True
    if load_model_from_mlflow():
        return True
    if load_model_from_local():
        return True
    return False

# Charger le modele au demarrage
ensure_model_loaded()

@app.route("/predict", methods=["POST"])
def predict():
    """Endpoint de prediction."""
    start_time = time.time()
    
    if not ensure_model_loaded():
        return jsonify({"error": "Aucun pipeline disponible"}), 503

    try:
        data = request.get_json(force=True)
        
        # Validation JSON (Format du TP)
        if "features" in data:
            features_raw = np.array(data["features"]).reshape(1, -1)
        elif "instances" in data:
            features_raw = np.array(data["instances"])
        else:
            return jsonify({"error": "Format invalide. Utilisez 'features' ou 'instances'."}), 400

        # --- LE PIPELINE XGBOOST ---
        X_eng = engineer(features_raw, FEATURE_NAMES)
        X_clust = add_clusters(X_eng, _km)
        X_final = _scaler.transform(X_clust)

        predictions = _model.predict(X_final).tolist()
        
        # Format de retour demandé par le TP
        latency_ms = (time.time() - start_time) * 1000
        response = {
            "predictions": [round(float(p), 4) for p in predictions],
            "model_source": model_source,
            "latency_ms": round(latency_ms, 2),
            "n_samples": len(predictions)
        }
        
        logger.info(f"Prediction reussie : latence={latency_ms:.2f}ms, source={model_source}")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Erreur de prediction : {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    """Endpoint de sante."""
    return jsonify({
        "status": "healthy",
        "model_loaded": _model is not None,
        "model_source": model_source
    }), 200

@app.route("/reload", methods=["POST"])
def reload_model():
    """Endpoint pour forcer le rechargement du modele depuis MLflow."""
    global _model
    _model = None # Reset
    if load_model_from_mlflow():
        return jsonify({"status": "reloaded", "model_source": model_source}), 200
    elif load_model_from_local():
        return jsonify({"status": "reloaded_from_local", "model_source": model_source}), 200
    else:
        return jsonify({"status": "error", "message": "Aucun modele disponible"}), 503

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)