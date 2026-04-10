# ml/train.py
import os
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib

def load_data():
    housing = fetch_california_housing()
    return housing.data, housing.target, housing.feature_names

def engineer(X, names):
    df = {n: X[:, i] for i, n in enumerate(names)}
    extra = np.column_stack([
        df["AveRooms"] / (df["AveOccup"] + 1e-5),
        df["AveBedrms"] / (df["AveRooms"] + 1e-5),
        df["Population"] / (df["AveOccup"] + 1e-5),
        df["MedInc"] / (df["AveRooms"] + 1e-5),
        df["Latitude"] * df["Longitude"],
        np.sqrt((df["Latitude"] - 36.7)**2 + (df["Longitude"] - (-119.4))**2),
        np.log1p(df["Population"]),
        np.log1p(df["AveOccup"]),
        np.log1p(df["MedInc"]),
        df["MedInc"] ** 2,
        df["AveOccup"] ** 2
    ])
    return np.hstack([X, extra])

def add_clusters(X, kmeans_model):
    coords = X[:, 6:8]
    clusters = kmeans_model.predict(coords).reshape(-1, 1)
    dists = kmeans_model.transform(coords).min(axis=1).reshape(-1, 1)
    return np.hstack([X, clusters, dists])

def main():
    print("=" * 60)
    print("CloudPredict - Entrainement du modele")
    print("=" * 60)
    
    # [1/4] Chargement
    print("\n[1/4] Chargement du dataset California Housing...")
    X_raw, y, names = load_data()
    print(f" - Nombre d'echantillons : {len(X_raw)}")
    print(f" - Nombre de features : {len(names)}")
    print(f" - Features : {list(names)}")
    
    # Engineering interne (invisible dans les logs comme dans ton exemple)
    X_eng = engineer(X_raw, names)
    
    # [2/4] Split
    print("\n[2/4] Division train/test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X_eng, y, test_size=0.2, random_state=42)
    print(f" - Train : {len(X_train)} echantillons")
    print(f" - Test : {len(X_test)} echantillons")
    
    # [3/4] Entrainement (incluant Clustering + Scaling interne)
    print("\n[3/4] Entrainement du XGBoostRegressor...")
    km = KMeans(n_clusters=30, random_state=42, n_init=10)
    km.fit(X_train[:, 6:8])
    X_train_clust = add_clusters(X_train, km)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_clust)
    
    best_params = {
        'n_estimators': 2684,
        'learning_rate': 0.023463258960422607,
        'max_depth': 6,
        'min_child_weight': 3,
        'subsample': 0.909388760000992,
        'colsample_bytree': 0.8442245426717814,
        'gamma': 0.0001307453591362648,
        'reg_alpha': 0.5844805204033486,
        'reg_lambda': 2.0006219005796497,
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'rmse'
    }
    model = xgb.XGBRegressor(**best_params)
    model.fit(X_train_s, y_train, verbose=False)
    print(" - Entrainement termine !")
    
    # [4/4] Evaluation
    print("\n[4/4] Evaluation du modele...")
    # Préparation du test set avec les mêmes outils
    X_test_clust = add_clusters(X_test, km)
    X_test_s = scaler.transform(X_test_clust)
    y_pred = model.predict(X_test_s)
    
    print(f" - MAE : {mean_absolute_error(y_test, y_pred):.4f}")
    print(f" - RMSE : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    print(f" - R2 : {r2_score(y_test, y_pred):.4f}")
    
    # Sauvegarde des artefacts
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, "xgb_model.joblib")
    joblib.dump(model, model_path)
    joblib.dump(scaler, os.path.join(models_dir, "scaler.joblib"))
    joblib.dump(km, os.path.join(models_dir, "kmeans.joblib"))
    
    print(f"Modele sauvegarde dans : ml/models/xgb_model.joblib")
    
    print("\n" + "=" * 60)
    print("Entrainement termine avec succes !")
    print("=" * 60)

if __name__ == "__main__":
    main()