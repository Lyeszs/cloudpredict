# app/main.py
import time
import logging
from flask import Flask, request, jsonify, Response
# On importe nos fonctions personnalisées
from app.predict import predict, get_pipeline, FEATURE_NAMES
from app.metrics import (
    prediction_count,
    prediction_latency,
    prediction_errors,
    get_metrics
)
from app.logging_config import setup_logging

# 1. Initialiser le logging JSON
logger = setup_logging()

# 2. Créer l'application Flask
app = Flask(__name__)

@app.before_request
def log_request():
    """Log chaque requete entrante en JSON."""
    logger.info(f"Requete: {request.method} {request.path}")

@app.route("/health", methods=["GET"])
def health():
    """Endpoint de health check pour Kubernetes/Docker."""
    try:
        # On vérifie que le trio Model/Scaler/KMeans est chargé
        get_pipeline()
        model_status = "loaded"
    except Exception as e:
        model_status = f"error: {str(e)}"
    
    response = {
        "status": "ok",
        "service": "cloudpredict-api",
        "version": "1.0.0",
        "model_status": model_status
    }
    status_code = 200 if model_status == "loaded" else 503
    return jsonify(response), status_code

@app.route("/metrics", methods=["GET"])
def metrics():
    """Endpoint pour que Prometheus vienne 'aspirer' les chiffres."""
    metrics_data, content_type = get_metrics()
    return Response(metrics_data, mimetype=content_type)

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    """L'endpoint principal de prédiction immobilière."""
    start_time = time.time()
    try:
        if not request.is_json:
            prediction_errors.labels(error_type="invalid_content_type").inc()
            return jsonify({"error": "Content-Type doit etre application/json"}), 415

        data = request.get_json()
        
        # Appel de notre fonction de prédiction optimisée (XGBoost + Transformation)
        result, errors = predict(data)
        
        if errors:
            prediction_errors.labels(error_type="validation_error").inc()
            prediction_count.labels(status="error").inc()
            return jsonify({"error": "Erreur de validation", "details": errors}), 422

        # Succès : Enregistrement des métriques
        latency = time.time() - start_time
        prediction_latency.observe(latency)
        prediction_count.labels(status="success").inc()
        
        result["latency_ms"] = round(latency * 1000, 2)
        logger.info(f"Prediction reussie: {result['prediction']}", extra={"extra_data": {"latency": result["latency_ms"]}})
        
        return jsonify(result), 200

    except Exception as e:
        prediction_errors.labels(error_type="internal_error").inc()
        prediction_count.labels(status="error").inc()
        logger.error(f"Erreur interne: {str(e)}", exc_info=True)
        return jsonify({"error": "Erreur interne du serveur", "message": str(e)}), 500

@app.route("/info", methods=["GET"])
def info():
    """Documentation dynamique de l'API."""
    return jsonify({
        "service": "cloudpredict-api",
        "version": "1.0.0",
        "model": "XGBoostRegressor (Optimized)",
        "dataset": "California Housing",
        "features": FEATURE_NAMES,
        "endpoints": {
            "GET /health": "Etat de santé du service",
            "GET /metrics": "Metriques Prometheus",
            "POST /predict": "Estimation de prix (JSON)"
        }
    }), 200

@app.route("/", methods=["GET"])
def root():
    return jsonify({"message": "Bienvenue sur CloudPredict API", "documentation": "/info"}), 200

if __name__ == "__main__":
    # host="0.0.0.0" est CRITIQUE pour Docker
    app.run(host="0.0.0.0", port=5000, debug=True)