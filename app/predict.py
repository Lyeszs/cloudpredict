# app/predict.py
import os
import numpy as np
import joblib
import logging
# On importe tes fonctions de transformation depuis le dossier ml
from ml.train import engineer, add_clusters

logger = logging.getLogger("cloudpredict")

FEATURE_NAMES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms", 
    "Population", "AveOccup", "Latitude", "Longitude"
]

# On a besoin des 3 artefacts pour que ça marche
_model = None
_scaler = None
_km = None

def load_artifacts():
    """Charge le modèle, le scaler et le kmeans."""
    global _model, _scaler, _km
    
    # On définit où sont rangés les fichiers
    base_path = os.path.join(os.path.dirname(__file__), "..", "ml", "models")
    
    try:
        _model = joblib.load(os.path.join(base_path, "xgb_model.joblib"))
        _scaler = joblib.load(os.path.join(base_path, "scaler.joblib"))
        _km = joblib.load(os.path.join(base_path, "kmeans.joblib"))
        logger.info(f"Pipeline complet chargé (Model, Scaler, KMeans) depuis {base_path}")
    except Exception as e:
        logger.error(f"Erreur lors du chargement des artefacts : {str(e)}")
        raise FileNotFoundError("Impossible de charger le pipeline de prédiction.")

def get_pipeline():
    """Retourne les artefacts, en les chargeant si nécessaire."""
    if _model is None or _scaler is None or _km is None:
        load_artifacts()
    return _model, _scaler, _km

def validate_features(data):
    """Vérifie si les données envoyées par l'utilisateur sont correctes."""
    if "features" not in data:
        return None, ["Le champ 'features' est requis dans le body JSON."]
    
    features = data["features"]
    try:
        if isinstance(features, dict):
            missing = [f for f in FEATURE_NAMES if f not in features]
            if missing:
                return None, [f"Features manquantes : {missing}"]
            values = [float(features[f]) for f in FEATURE_NAMES]
        elif isinstance(features, list):
            if len(features) != len(FEATURE_NAMES):
                return None, [f"Attendu {len(FEATURE_NAMES)} features, reçu {len(features)}."]
            values = [float(v) for v in features]
        else:
            return None, ["Le champ 'features' doit être un dictionnaire ou une liste."]
        
        return np.array(values).reshape(1, -1), None
    except ValueError:
        return None, ["Toutes les features doivent être des nombres (float/int)."]

def predict(data):
    """Effectue la transformation complète et la prédiction."""
    # 1. Validation des entrées
    features_raw, errors = validate_features(data)
    if errors:
        return None, errors

    # 2. Récupération des outils
    model, scaler, km = get_pipeline()

    # 3. LE PIEUVRE DE TRANSFORMATION (Ton secret sauce)
    # A. Engineering (AveRooms/Occup, logs, etc.)
    X_eng = engineer(features_raw, FEATURE_NAMES)
    # B. Ajout des clusters géo via le KMeans chargé
    X_clust = add_clusters(X_eng, km)
    # C. Normalisation via le Scaler chargé
    X_final = scaler.transform(X_clust)

    # 4. Prédiction finale
    prediction = model.predict(X_final)
    
    return {
        "prediction": round(float(prediction[0]), 4),
        "unit": "100k USD",
        "description": "Prix median des maisons (en centaines de milliers de dollars)",
        "features_used": FEATURE_NAMES,
        "input_values": features_raw[0].tolist()
    }, None