# ml/test_pipeline.py
"""Script de test end-to-end du pipeline MLOps."""
import subprocess
import sys
import time
import requests
import json
import os

# On s'assure que le PYTHONPATH est correct pour les imports internes
os.environ["PYTHONPATH"] = "."

API_URL = "http://localhost:8000"
# On utilise l'IP Minikube pour MLflow car le script tourne sur ton hôte
MLFLOW_URL = "http://192.168.49.2:30500"

def run_step(name, command):
    """Execute une etape et affiche le resultat."""
    print(f"\n{'='*60}")
    print(f" ETAPE : {name}")
    print(f"{'='*60}")
    # On ajoute PYTHONPATH=. pour éviter les ModuleNotFoundError
    full_command = f"PYTHONPATH=. {command}"
    result = subprocess.run(full_command, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"[ERREUR] {result.stderr}")
        return False
    return True

def test_api(endpoint, method="GET", data=None):
    """Teste un endpoint de l'API."""
    try:
        if method == "GET":
            resp = requests.get(f"{API_URL}{endpoint}", timeout=10)
        else:
            resp = requests.post(f"{API_URL}{endpoint}", json=data, timeout=10)
        print(f" {method} {endpoint} -> {resp.status_code}")
        print(f" Reponse : {json.dumps(resp.json(), indent=2)}")
        return resp.status_code == 200
    except Exception as e:
        print(f" [ERREUR] {e}")
        return False

def main():
    print("=" * 60)
    print(" TEST END-TO-END DU PIPELINE MLOps")
    print("=" * 60)
    results = []

    # 1. Entrainement initial
    ok = run_step("Entrainement initial", f"{sys.executable} ml/train.py")
    results.append(("Entrainement", ok))

    # 2. Promotion en Production
    ok = run_step("Promotion en Production", f"{sys.executable} ml/promote_model.py production")
    results.append(("Promotion", ok))

    # 3. Test API - Health
    print(f"\n{'='*60}")
    print(f" ETAPE : Test API Health")
    print(f"{'='*60}")
    ok = test_api("/health")
    results.append(("API Health", ok))

    # 4. Test API - Prediction
    print(f"\n{'='*60}")
    print(f" ETAPE : Test API Prediction")
    print(f"{'='*60}")
    ok = test_api("/predict", method="POST", data={
        "features": [8.3252, 41.0, 6.984127, 1.023810, 322.0, 2.555556, 37.88, -122.23]
    })
    results.append(("API Predict", ok))

    # 5. Detection de drift (sans drift)
    ok = run_step("Detection de drift (normal)", f"{sys.executable} ml/drift_detector.py 0.0")
    results.append(("Drift normal", ok))

    # 6. Detection de drift (avec drift)
    ok = run_step("Detection de drift (simule)", f"{sys.executable} ml/drift_detector.py 1.0")
    results.append(("Drift simule", ok))

    # 7. Reentrainement force
    ok = run_step("Reentrainement", f"{sys.executable} ml/retrain.py --force")
    results.append(("Reentrainement", ok))

    # 8. Rechargement du modele
    print(f"\n{'='*60}")
    print(f" ETAPE : Rechargement du modele")
    print(f"{'='*60}")
    ok = test_api("/reload", method="POST")
    results.append(("Reload modele", ok))

    # Resume final
    print("\n" + "=" * 60)
    print(" RESUME DES TESTS")
    print("=" * 60)
    all_ok = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f" {name:<25} : {status}")
        if not ok:
            all_ok = False
    print("=" * 60)
    msg = "TOUS LES TESTS PASSENT" if all_ok else "CERTAINS TESTS ECHOUIENT"
    print(f" Resultat global : {msg}")
    print("=" * 60)

if __name__ == "__main__":
    main()