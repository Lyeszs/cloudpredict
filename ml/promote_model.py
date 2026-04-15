# ml/promote_model.py
"""
Script de promotion du modele dans le Model Registry MLflow.
Gere les transitions : None -> Staging -> Production
"""
import os
import sys
import mlflow
from mlflow.tracking import MlflowClient

# ---- Configuration ----
# On utilise l'URL Minikube qui fonctionne pour toi
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://192.168.49.2:30500")
MODEL_NAME = "housing-price-predictor"

def get_latest_model_version(client, model_name):
    """Recupere la derniere version du modele."""
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        print(f"[ERREUR] Aucun modele trouve avec le nom '{model_name}'")
        sys.exit(1)
    latest = max(versions, key=lambda v: int(v.version))
    print(f"[INFO] Derniere version trouvee : v{latest.version} (stage: {latest.current_stage})")
    return latest

def promote_to_staging(client, model_name, version):
    """Promouvoit un modele vers Staging."""
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Staging",
        archive_existing_versions=False
    )
    print(f"[INFO] Modele {model_name} v{version} promu vers Staging")

def promote_to_production(client, model_name, version):
    """Promouvoit un modele vers Production (archive l'ancien)."""
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production",
        archive_existing_versions=True
    )
    print(f"[INFO] Modele {model_name} v{version} promu vers Production")
    print("[INFO] Les anciennes versions Production ont ete archivees")

def list_all_versions(client, model_name):
    """Affiche toutes les versions du modele."""
    versions = client.search_model_versions(f"name='{model_name}'")
    print(f"\n{'Version':<10} {'Stage':<15} {'Run ID':<35} {'Status'}")
    print("-" * 75)
    for v in sorted(versions, key=lambda x: int(x.version)):
        print(f"v{v.version:<9} {v.current_stage:<15} {v.run_id:<35} {v.status}")
    print()

def main():
    """Promouvoir le dernier modele vers Staging puis Production."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    
    # Determiner l'action
    action = sys.argv[1] if len(sys.argv) > 1 else "production"
    print(f"[INFO] Connexion a MLflow : {MLFLOW_TRACKING_URI}")
    print(f"[INFO] Action demandee : {action}")
    
    # Lister les versions existantes
    list_all_versions(client, MODEL_NAME)
    
    # Recuperer la derniere version
    latest = get_latest_model_version(client, MODEL_NAME)
    
    if action == "staging":
        promote_to_staging(client, MODEL_NAME, latest.version)
    elif action == "production":
        # Passer par Staging d'abord si pas deja en Staging
        if latest.current_stage == "None":
            promote_to_staging(client, MODEL_NAME, latest.version)
        promote_to_production(client, MODEL_NAME, latest.version)
    elif action == "list":
        pass # Deja affiche ci-dessus
    else:
        print(f"[ERREUR] Action inconnue : {action}")
        print("Usage : python ml/promote_model.py [staging|production|list]")
        sys.exit(1)
        
    # Afficher l'etat final
    list_all_versions(client, MODEL_NAME)
    print("[INFO] Promotion terminee avec succes !")

if __name__ == "__main__":
    main()