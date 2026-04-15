# ============================================
# Stage 1 : Builder - Installer les dependances
# ============================================
FROM python:3.11-slim AS builder

WORKDIR /build

# Installation des outils de compilation nécessaires pour certaines libs (ex: XGBoost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copier et installer les dependances dans un dossier temporaire /install
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ============================================
# Stage 2 : Runtime - Image finale legere
# ============================================
FROM python:3.11-slim AS runtime

# Metadata
LABEL maintainer="CloudPredict Team"
LABEL description="CloudPredict ML Prediction API"
LABEL version="1.0.0"

# Variables d'environnement
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# Note : Ton code actuel charge 3 modèles, MODEL_PATH ici est indicatif
ENV MODEL_PATH=/app/ml/models/xgb_model.joblib
ENV FLASK_APP=app.main
ENV FLASK_ENV=production

# Sécurité : Créer un utilisateur non-root pour ne pas executer l'app en tant qu'admin
RUN groupadd -r cloudpredict && useradd -r -g cloudpredict cloudpredict

WORKDIR /app

# Récupérer uniquement le nécessaire du builder (image finale plus légère)
COPY --from=builder /install /usr/local

# Copier le code source avec les bons droits
COPY --chown=cloudpredict:cloudpredict app/ ./app/
COPY --chown=cloudpredict:cloudpredict ml/ ./ml/

# Créer le dossier pour les modeles
RUN mkdir -p /app/ml/models && chown -R cloudpredict:cloudpredict /app

# Passer a l'utilisateur non-root
USER cloudpredict

# Port exposé
EXPOSE 5000

# Health check : Le système vérifie toutes les 30s si l'API répond 200 OK
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health').getcode() == 200 or exit(1)"

# Lancer avec gunicorn (serveur de production) au lieu du serveur de dev Flask
# On utilise 2 workers et 2 threads pour gérer les requêtes en parallèle
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--threads", "2", "--timeout", "120", "app.main:app"]