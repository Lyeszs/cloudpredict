# tests/test_predict.py
import pytest
import numpy as np
from app.predict import validate_features, predict, get_model, FEATURE_NAMES

class TestValidateFeatures:
    """Tests pour la validation des features."""

    def test_valid_dict_features(self):
        """Un dictionnaire complet de features doit etre valide."""
        data = {
            "features": {
                "MedInc": 8.3252,
                "HouseAge": 41.0,
                "AveRooms": 6.984,
                "AveBedrms": 1.024,
                "Population": 322.0,
                "AveOccup": 2.556,
                "Latitude": 37.88,
                "Longitude": -122.23
            }
        }
        features_array, errors = validate_features(data)
        assert errors is None
        assert features_array is not None
        assert features_array.shape == (1, 8)

    def test_valid_list_features(self):
        """Une liste de 8 valeurs doit etre valide."""
        data = {
            "features": [8.3252, 41.0, 6.984, 1.024, 322.0, 2.556, 37.88, -122.23]
        }
        features_array, errors = validate_features(data)
        assert errors is None
        assert features_array is not None
        assert features_array.shape == (1, 8)

    def test_missing_features_key(self):
        """L'absence de la cle 'features' doit retourner une erreur."""
        data = {"data": [1, 2, 3]}
        features_array, errors = validate_features(data)
        assert features_array is None
        assert errors is not None
        assert len(errors) > 0

    def test_wrong_list_length(self):
        """Une liste de mauvaise longueur doit retourner une erreur."""
        data = {"features": [1.0, 2.0, 3.0]}
        features_array, errors = validate_features(data)
        assert features_array is None
        assert errors is not None

    def test_missing_dict_keys(self):
        """Un dictionnaire incomplet doit retourner une erreur."""
        data = {"features": {"MedInc": 8.0, "HouseAge": 41.0}}
        features_array, errors = validate_features(data)
        assert features_array is None
        assert errors is not None

    def test_invalid_features_type(self):
        """Un type invalide pour features doit retourner une erreur."""
        data = {"features": "invalid"}
        features_array, errors = validate_features(data)
        assert features_array is None
        assert errors is not None

class TestPredict:
    """Tests pour la fonction de prediction."""

    def test_predict_returns_result(self):
        """La prediction doit retourner un resultat valide."""
        data = {
            "features": [8.3252, 41.0, 6.984, 1.024, 322.0, 2.556, 37.88, -122.23]
        }
        result, errors = predict(data)
        assert errors is None
        assert result is not None
        assert "prediction" in result
        assert isinstance(result["prediction"], float)
        assert result["prediction"] > 0

    def test_predict_result_format(self):
        """Le resultat de prediction doit avoir le bon format."""
        data = {
            "features": [5.0, 30.0, 5.5, 1.1, 1200.0, 3.0, 34.05, -118.25]
        }
        result, errors = predict(data)
        assert errors is None
        assert "unit" in result
        assert "description" in result
        assert "features_used" in result
        assert "input_values" in result
        assert len(result["features_used"]) == 8
        assert len(result["input_values"]) == 8

    def test_predict_with_invalid_data(self):
        """Des donnees invalides doivent retourner des erreurs."""
        data = {"features": [1.0]}
        result, errors = predict(data)
        assert result is None
        assert errors is not None

class TestGetModel:
    """Tests pour le chargement du modele."""

    def test_model_loads_successfully(self):
        model = get_model()
        assert model is not None

    def test_model_has_predict_method(self):
        model = get_model()
        assert hasattr(model, "predict")

    # CETTE FONCTION DOIT ETRE INDENTÉE ICI, DANS LA CLASSE
    def test_model_predicts_array(self):
        """Le modele doit pouvoir predire a partir d'un array numpy."""
        from app.predict import get_pipeline, FEATURE_NAMES
        from ml.train import engineer, add_clusters
        
        model, scaler, km = get_pipeline()
        X_raw = np.array([[8.3252, 41.0, 6.984, 1.024, 322.0, 2.556, 37.88, -122.23]])
        
        # Transformation obligatoire pour passer de 8 à 21 colonnes
        X_eng = engineer(X_raw, FEATURE_NAMES)
        X_clust = add_clusters(X_eng, km)
        X_final = scaler.transform(X_clust)
        
        prediction = model.predict(X_final)
        assert prediction is not None
        assert prediction[0] > 0

class TestFeatureNames:
    """Tests pour les noms de features."""

    def test_feature_names_count(self):
        """Il doit y avoir exactement 8 features."""
        assert len(FEATURE_NAMES) == 8

    def test_feature_names_content(self):
        """Les noms de features doivent correspondre au dataset California Housing."""
        expected = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
                    "Population", "AveOccup", "Latitude", "Longitude"]
        assert FEATURE_NAMES == expected






