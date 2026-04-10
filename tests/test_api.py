# tests/test_api.py
import json
import pytest
from app.main import app

@pytest.fixture
def client():
    """Crée un client de test Flask."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

class TestHealthEndpoint:
    """Tests pour l'endpoint /health."""
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_json(self, client):
        response = client.get("/health")
        data = response.get_json()
        assert data is not None
        assert data["status"] == "ok"
        assert "model_status" in data

class TestPredictEndpoint:
    """Tests pour l'endpoint /predict."""
    def test_predict_with_dict_features(self, client):
        """Vérifie qu'un dictionnaire de features fonctionne."""
        payload = {
            "features": {
                "MedInc": 8.32, "HouseAge": 41.0, "AveRooms": 6.98,
                "AveBedrms": 1.02, "Population": 322.0, "AveOccup": 2.55,
                "Latitude": 37.88, "Longitude": -122.23
            }
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.get_json()
        assert "prediction" in data
        assert isinstance(data["prediction"], float)

    def test_predict_with_list_features(self, client):
        """Vérifie qu'une liste de features fonctionne."""
        payload = {"features": [8.3, 41.0, 6.9, 1.0, 322.0, 2.5, 37.8, -122.2]}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

    def test_predict_missing_features(self, client):
        """Une requête vide doit renvoyer une erreur 422."""
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_predict_wrong_content_type(self, client):
        """L'envoi de texte brut doit renvoyer une erreur 415."""
        response = client.post("/predict", data="not json", content_type="text/plain")
        assert response.status_code == 415

class TestMetricsEndpoint:
    """Tests pour l'endpoint /metrics."""
    def test_metrics_contains_data(self, client):
        # On fait une prédiction pour générer de la donnée
        client.post("/predict", json={"features": [1,2,3,4,5,6,7,8]})
        
        response = client.get("/metrics")
        content = response.data.decode("utf-8")
        # On vérifie les noms de tes métriques réelles
        assert "prediction_requests_total" in content
        assert "prediction_duration_seconds" in content

class TestInfoEndpoint:
    """Tests pour l'endpoint /info."""
    def test_info_content(self, client):
        response = client.get("/info")
        data = response.get_json()
        assert response.status_code == 200
        assert len(data["features"]) == 8