1. Architecture (0:00 - 1:30)

Support : Le slide basé sur l'image image_40a38e.png.

    Discours : "Mon projet CloudPredict repose sur un cluster Kubernetes orchestrant une pile MLOps complète. On y trouve une API Flask pour les prédictions, une base PostgreSQL pour la persistance, et MLflow comme cerveau central pour le tracking et le registre de modèles. Le tout est surveillé par Prometheus, Grafana et Loki pour l'observabilité."

2. Kubernetes (1:30 - 3:00)

Objectif : Prouver que l'infrastructure est saine.

    Commandes à lancer :
    Bash

    # Afficher tous les pods en cours d'exécution
    kubectl get pods -n default

    # Montrer les services (NodePorts)
    kubectl get svc

    À dire : "Tous les composants sont conteneurisés et déployés via des objets Deployment et Service. Le cluster est stable, aucun pod n'est en erreur."

3. API de prédiction (3:00 - 4:30)

Objectif : Montrer que l'API est liée à MLflow.

    Commandes à lancer :
    Bash

    # Test de santé (doit montrer model_source: mlflow)
    curl http://localhost:8000/health

    # Test de prédiction
    curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"features": [8.3252, 41.0, 6.984127, 1.023810, 322.0, 2.555556, 37.88, -122.23]}'

    À dire : "L'API ne se contente pas de prédire ; elle récupère dynamiquement le modèle marqué en 'Production' dans le registre MLflow."

4. MLflow (4:30 - 6:00)

Objectif : Montrer la traçabilité.

    Action : Ouvre http://192.168.49.2:30500 dans le navigateur.

    À montrer :

        L'expérience cloudpredict-xgboost avec l'historique des runs.

        L'onglet Models : Montrer que la v5 est en Production et les anciennes en Archived.

5. Monitoring (6:00 - 7:30)

Objectif : Montrer l'observabilité.

    Action : Ouvre Grafana (port 3000 ou 30300).

    À montrer : Les courbes de latence de l'API et l'utilisation CPU/RAM des pods.

    À dire : "Nous monitorons non seulement la santé du cluster, mais aussi les métriques métier via Prometheus."

6. Drift + Retrain (7:30 - 9:00)

Objectif : Démontrer l'automatisation (la partie la plus valorisée).

    Commandes à lancer :
    Bash

    # 1. Simuler un drift fort
    PYTHONPATH=. python ml/drift_detector.py 1.0

    # 2. Ouvrir le rapport HTML dans ml/reports/ (Evidently)

    # 3. Lancer le réentraînement forcé (Etape 6)
    PYTHONPATH=. python ml/retrain.py --force

    À dire : "Si une dérive des données est détectée par Evidently, un pipeline de réentraînement automatique est déclenché. Le nouveau modèle est évalué puis promu en Production sans intervention humaine."

7. Conclusion (9:00 - 10:00)

Objectif : Synthèse et perspectives.

    Points forts : Pipeline end-to-end automatisé , haute disponibilité grâce à Kubernetes, traçabilité complète via MLflow.

    Améliorations : Implémenter des tests A/B avant la promotion, utiliser S3 pour le stockage des artefacts au lieu d'un PVC local.

💡 Commandes de secours (si ça bloque)

    Relancer le port-forward MLflow : kubectl port-forward svc/mlflow-service 5000:5000.

    Forcer le rechargement du modèle dans l'API : curl -X POST http://localhost:8000/reload.

    Vérifier les logs du réentraînement : kubectl logs -l component=retrain.












1. Les variables d'environnement (PowerShell vs Bash)

Sur Linux, tu utilisais VARIABLE=valeur commande. Sur Windows (PowerShell), cela ne fonctionne pas.

    Linux (Ce que tu faisais) :
    PYTHONPATH=. MLFLOW_TRACKING_URI="http://localhost:5000" python ml/train.py

    Windows (PowerShell - Ce qu'il faut faire) :
    PowerShell

    $env:PYTHONPATH="."; $env:MLFLOW_TRACKING_URI="http://localhost:5000"; python ml/train.py

    Alternative : Si tu utilises le terminal Git Bash sur Windows, tu peux garder tes commandes Linux habituelles.

2. Les Slats et Chemins (Slashes)

Linux utilise des slashs / alors que Windows utilise des backslashs \.

    Python gère généralement bien les / dans le code, mais dans ton terminal PowerShell, fais attention aux chemins de fichiers.

    Vérification MLflow : Assure-toi que mlflow est bien dans ton PATH Windows. Teste avec mlflow --version dès ton arrivée en classe.

3. Accès au Cluster (Minikube / Docker Desktop)

L'adresse IP pour accéder à MLflow change selon l'installation sur Windows :

    Docker Desktop : MLflow sera probablement sur http://localhost:30500.

    Minikube : Tu devras récupérer l'URL spécifique avec la commande :

    minikube service mlflow-service --url.

4. Le Port-Forward (Ton filet de sécurité)

Si le réseau de l'école bloque les NodePorts (30500), le port-forward est plus stable sur Windows pour faire le lien entre ton code et le cluster:
PowerShell

kubectl port-forward svc/mlflow-service 5000:5000

Laisse ce terminal ouvert, et dans ton code/scripts, utilise http://localhost:5000.
Résumé des commandes à adapter pour Windows
Action	Commande Windows (PowerShell)
Lancer l'API	$env:PYTHONPATH="."; $env:MLFLOW_TRACKING_URI="http://localhost:5000"; python app/predict.py
Test de Drift	$env:PYTHONPATH="."; python ml/drift_detector.py 1.0
Réentraînement	$env:PYTHONPATH="."; python ml/retrain.py --force
Curl (Prédiction)	Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post -Body '{"features": [8.3, 41.0, 6.9, 1.0, 322.0, 2.5, 37.8, -122.2]}' -ContentType "application/json"