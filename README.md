Creation de l’arboressence  
CLOUDPREDICT -> (app,docker,k8s,ml/models,monitoring,tests) 

Creation du venv et de requirements.txt avec les librairie demand 

Modele de ML  

Pour l’instant XGBOOST  

===================================  

R² Score : 0.8667  

MAE : 0.2701  

Temps : 5.15s  

=================================== 

 

Integration de joblib : 

Kmeans.joblib pour le feature ingeniering pour les donnée geographique surtout

scaler.joblib pour les echelle changé

xgb_model.joblib pour le modele finale avec les 21 variable et les methode de tunning adapté



http://192.168.49.2:30500/#/experiments/2/runs/e9169da5e00841b4bfa60277a43c5e08 
(url pour mlflow)

train sur mon pc linux 

(venv) lyes@lyes-MS-7C96:~/cloudpredict$ python ml/train.py
============================================================
CloudPredict - Entrainement du modele XGBoost + MLflow
============================================================

[1/4] Chargement du dataset...

[2/4] Division train/test...
[INFO] Donnees de reference sauvegardees dans ml/reference_data.csv

[3/4] Entrainement du XGBoostRegressor...
 - Entrainement termine !

[4/4] Evaluation du modele...
 - MAE : 0.2700
 - RMSE : 0.4179
 - R2 : 0.8667
/home/lyes/cloudpredict/venv/lib/python3.12/site-packages/xgboost/sklearn.py:1116: UserWarning: [00:12:39] WARNING: /__w/xgboost/xgboost/src/c_api/c_api.cc:1573: Saving model in the UBJSON format as default.  You can use a file extension: `json` or `ubj` to choose between formats.
  self.get_booster().save_model(fname)
/home/lyes/cloudpredict/venv/lib/python3.12/site-packages/xgboost/sklearn.py:1125: UserWarning: [00:12:42] WARNING: /__w/xgboost/xgboost/src/c_api/c_api.cc:1509: Unknown file format: `xgb`. Using UBJSON (`ubj`) as a guess.
  self.get_booster().load_model(fname)
Registered model 'housing-price-predictor' already exists. Creating a new version of this model...
2026/04/16 00:12:43 INFO mlflow.store.model_registry.abstract_store: Waiting up to 300 seconds for model version to finish creation. Model name: housing-price-predictor, version 2
Created version '2' of model 'housing-price-predictor'.
[INFO] Modele enregistre dans le Model Registry : housing-price-predictor
[INFO] MLflow Run ID : 85f335671c8d4962ad06075a00d416f7
🏃 View run training-xgb-run at: http://192.168.49.2:30500/#/experiments/2/runs/85f335671c8d4962ad06075a00d416f7
🧪 View experiment at: http://192.168.49.2:30500/#/experiments/2

============================================================
Pipeline termine et donnees envoyees a MLflow !
============================================================
(venv) lyes@lyes-MS-7C96:~/cloudpredict$ python ml/promote_model.py production
[INFO] Connexion a MLflow : http://192.168.49.2:30500
[INFO] Action demandee : production

Version    Stage           Run ID                              Status
---------------------------------------------------------------------------
v1         Production      e9169da5e00841b4bfa60277a43c5e08    READY
v2         None            85f335671c8d4962ad06075a00d416f7    READY

[INFO] Derniere version trouvee : v2 (stage: None)
/home/lyes/cloudpredict/ml/promote_model.py:28: FutureWarning: ``mlflow.tracking.client.MlflowClient.transition_model_version_stage`` is deprecated since 2.9.0. Model registry stages will be removed in a future major release. To learn more about the deprecation of model registry stages, see our migration guide here: https://mlflow.org/docs/latest/model-registry.html#migrating-from-stages
  client.transition_model_version_stage(
[INFO] Modele housing-price-predictor v2 promu vers Staging
/home/lyes/cloudpredict/ml/promote_model.py:38: FutureWarning: ``mlflow.tracking.client.MlflowClient.transition_model_version_stage`` is deprecated since 2.9.0. Model registry stages will be removed in a future major release. To learn more about the deprecation of model registry stages, see our migration guide here: https://mlflow.org/docs/latest/model-registry.html#migrating-from-stages
  client.transition_model_version_stage(
[INFO] Modele housing-price-predictor v2 promu vers Production
[INFO] Les anciennes versions Production ont ete archivees

Version    Stage           Run ID                              Status
---------------------------------------------------------------------------
v1         Archived        e9169da5e00841b4bfa60277a43c5e08    READY
v2         Production      85f335671c8d4962ad06075a00d416f7    READY

[INFO] Promotion terminee avec succes !
(venv) lyes@lyes-MS-7C96:~/cloudpredict$ curl http://localhost:8000/health
# Envoyer une prediction (valeurs typiques de la Californie)
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{"features": [8.3252, 41.0, 6.984127, 1.023810, 322.0, 2.555556, 37.88, -122.23]}'
# Envoyer plusieurs predictions
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{"instances": [
[8.3252, 41.0, 6.984127, 1.023810, 322.0, 2.555556, 37.88, -122.23],
[3.5, 25.0, 5.5, 1.1, 1500.0, 3.2, 34.05, -118.25],
[5.1, 30.0, 6.0, 1.05, 800.0, 2.8, 37.55, -122.0]
]}'
{"model_loaded":true,"model_source":"mlflow:housing-price-predictor/Production","status":"healthy"}
{"latency_ms":4.26,"model_source":"mlflow:housing-price-predictor/Production","n_samples":1,"predictions":[4.3045]}
{"latency_ms":4.09,"model_source":"mlflow:housing-price-predictor/Production","n_samples":3,"predictions":[4.3045,2.0125,2.5301]}
(venv) lyes@lyes-MS-7C96:~/cloudpredict$ python ml/drift_detector.py 0.0
============================================================
 DETECTION DE DRIFT 
============================================================
[INFO] Donnees de reference chargees (16512 lignes)
[INFO] Execution de la detection de drift Evidently...
[INFO] Rapport HTML sauvegarde : ml/reports/drift_report_20260416_001415.html
[INFO] Metriques de drift enregistrees dans MLflow
🏃 View run drift-detection at: http://192.168.49.2:30500/#/experiments/2/runs/29fe2b79fd3f4bfab2000eafdf9ce834
🧪 View experiment at: http://192.168.49.2:30500/#/experiments/2

 RESULTATS :
 Drift detecte : NON
 Part de drift : 0.00%
 Colonnes en drift : 0/8
(venv) lyes@lyes-MS-7C96:~/cloudpredict$ python ml/drift_detector.py 1.0
============================================================
 DETECTION DE DRIFT 
============================================================
[INFO] Donnees de reference chargees (16512 lignes)
[INFO] Application de drift simule (niveau: 1.0)
[INFO] Execution de la detection de drift Evidently...
[INFO] Rapport HTML sauvegarde : ml/reports/drift_report_20260416_001428.html
[INFO] Metriques de drift enregistrees dans MLflow
🏃 View run drift-detection at: http://192.168.49.2:30500/#/experiments/2/runs/91ad7204c815405cbb340baca6d73b62
🧪 View experiment at: http://192.168.49.2:30500/#/experiments/2

 RESULTATS :
 Drift detecte : OUI
 Part de drift : 0.00%
 Colonnes en drift : 4/8
(venv) lyes@lyes-MS-7C96:~/cloudpredict$ python ml/retrain.py
Traceback (most recent call last):
  File "/home/lyes/cloudpredict/ml/retrain.py", line 24, in <module>
    from ml.train import engineer, add_clusters
ModuleNotFoundError: No module named 'ml'
(venv) lyes@lyes-MS-7C96:~/cloudpredict$ 

e l'etape ou faut reantrainer avec retrain 
PYTHONPATH=. python ml/retrain.py


pipelin ede test complete 

(venv) lyes@lyes-MS-7C96:~/cloudpredict$ python ml/test_pipeline.py
============================================================
 TEST END-TO-END DU PIPELINE MLOps
============================================================

============================================================
 ETAPE : Entrainement initial
============================================================
============================================================
CloudPredict - Entrainement du modele XGBoost + MLflow
============================================================

[1/4] Chargement du dataset...

[2/4] Division train/test...
[INFO] Donnees de reference sauvegardees dans ml/reference_data.csv

[3/4] Entrainement du XGBoostRegressor...
 - Entrainement termine !

[4/4] Evaluation du modele...
 - MAE : 0.2700
 - RMSE : 0.4179
 - R2 : 0.8667
[INFO] Modele enregistre dans le Model Registry : housing-price-predictor
[INFO] MLflow Run ID : 9a13bec5eb644d369fef1bdfacbb2d7c
🏃 View run training-xgb-run at: http://192.168.49.2:30500/#/experiments/2/runs/9a13bec5eb644d369fef1bdfacbb2d7c
🧪 View experiment at: http://192.168.49.2:30500/#/experiments/2

============================================================
Pipeline termine et donnees envoyees a MLflow !
============================================================


============================================================
 ETAPE : Promotion en Production
============================================================
[INFO] Connexion a MLflow : http://192.168.49.2:30500
[INFO] Action demandee : production

Version    Stage           Run ID                              Status
---------------------------------------------------------------------------
v1         Archived        e9169da5e00841b4bfa60277a43c5e08    READY
v2         Archived        85f335671c8d4962ad06075a00d416f7    READY
v3         Production      e6525d90d3e645ad81e62eddf0b89871    READY
v4         None            9a13bec5eb644d369fef1bdfacbb2d7c    READY

[INFO] Derniere version trouvee : v4 (stage: None)
[INFO] Modele housing-price-predictor v4 promu vers Staging
[INFO] Modele housing-price-predictor v4 promu vers Production
[INFO] Les anciennes versions Production ont ete archivees

Version    Stage           Run ID                              Status
---------------------------------------------------------------------------
v1         Archived        e9169da5e00841b4bfa60277a43c5e08    READY
v2         Archived        85f335671c8d4962ad06075a00d416f7    READY
v3         Archived        e6525d90d3e645ad81e62eddf0b89871    READY
v4         Production      9a13bec5eb644d369fef1bdfacbb2d7c    READY

[INFO] Promotion terminee avec succes !


============================================================
 ETAPE : Test API Health
============================================================
 GET /health -> 200
 Reponse : {
  "model_loaded": true,
  "model_source": "mlflow:housing-price-predictor/Production",
  "status": "healthy"
}

============================================================
 ETAPE : Test API Prediction
============================================================
 POST /predict -> 200
 Reponse : {
  "latency_ms": 2.76,
  "model_source": "mlflow:housing-price-predictor/Production",
  "n_samples": 1,
  "predictions": [
    4.4106
  ]
}

============================================================
 ETAPE : Detection de drift (normal)
============================================================
============================================================
 DETECTION DE DRIFT 
============================================================
[INFO] Donnees de reference chargees (16512 lignes)
[INFO] Execution de la detection de drift Evidently...
[INFO] Rapport HTML sauvegarde : ml/reports/drift_report_20260416_001936.html
[INFO] Metriques de drift enregistrees dans MLflow
🏃 View run drift-detection at: http://192.168.49.2:30500/#/experiments/2/runs/d204903376774d48b9cd7d274c29dac8
🧪 View experiment at: http://192.168.49.2:30500/#/experiments/2

 RESULTATS :
 Drift detecte : NON
 Part de drift : 0.00%
 Colonnes en drift : 0/8


============================================================
 ETAPE : Detection de drift (simule)
============================================================
============================================================
 DETECTION DE DRIFT 
============================================================
[INFO] Donnees de reference chargees (16512 lignes)
[INFO] Application de drift simule (niveau: 1.0)
[INFO] Execution de la detection de drift Evidently...
[INFO] Rapport HTML sauvegarde : ml/reports/drift_report_20260416_001940.html
[INFO] Metriques de drift enregistrees dans MLflow
🏃 View run drift-detection at: http://192.168.49.2:30500/#/experiments/2/runs/e1ef100e7ab34974b8fe1549baf18d57
🧪 View experiment at: http://192.168.49.2:30500/#/experiments/2

 RESULTATS :
 Drift detecte : OUI
 Part de drift : 0.00%
 Colonnes en drift : 4/8


============================================================
 ETAPE : Reentrainement
============================================================
============================================================
 PIPELINE DE REENTRAINEMENT 
============================================================
[ETAPE 1] Verification du drift...
[INFO] Donnees de reference chargees (16512 lignes)
[INFO] Application de drift simule (niveau: 1.0)
[INFO] Execution de la detection de drift Evidently...
[INFO] Rapport HTML sauvegarde : ml/reports/drift_report_20260416_001944.html
[INFO] Drift detecte : OUI
[INFO] Modele Production actuel : v4 | R2: 0.8667
[ETAPE 2] Reentrainement du pipeline complet (XGBoost)...
[INFO] Nouvelles metriques -> MAE: 0.2741, R2: 0.8609
[INFO] Performance validee (diff R2: -0.0058) -> PROMOTION
[ETAPE 3] Enregistrement dans MLflow et Promotion...
🏃 View run retrain-20260416-0019 at: http://192.168.49.2:30500/#/experiments/2/runs/277b7e2a0edf4333ba24153b6b3c791e
🧪 View experiment at: http://192.168.49.2:30500/#/experiments/2
[INFO] Nouveau modele (v5) promu en Production !
[INFO] L'ancienne version a ete archivee.


============================================================
 ETAPE : Rechargement du modele
============================================================
 POST /reload -> 200
 Reponse : {
  "model_source": "mlflow:housing-price-predictor/Production",
  "status": "reloaded"
}

============================================================
 RESUME DES TESTS
============================================================
 Entrainement              : PASS
 Promotion                 : PASS
 API Health                : PASS
 API Predict               : PASS
 Drift normal              : PASS
 Drift simule              : PASS
 Reentrainement            : PASS
 Reload modele             : PASS
============================================================
 Resultat global : TOUS LES TESTS PASSENT
============================================================
(venv) lyes@lyes-MS-7C96:~/cloudpredict$ 
