# app/metrics.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# On utilise les noms exacts attendus par ton main.py (en minuscules)
prediction_count = Counter(
    'prediction_requests_total', 
    'Nombre total de requetes',
    ['status']
)

prediction_errors = Counter(
    'prediction_errors_total',
    'Nombre total d\'erreurs',
    ['error_type']
)

prediction_latency = Histogram(
    'prediction_duration_seconds', 
    'Temps de reponse de la prediction'
)

# Cette fonction est indispensable pour l'endpoint /metrics
def get_metrics():
    return generate_latest(), CONTENT_TYPE_LATEST