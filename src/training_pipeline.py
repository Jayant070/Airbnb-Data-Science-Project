from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scripts.training.price_model_trainer import (
    metrics_to_frame,
    train_decision_tree_regressor,
    train_knn_regressor,
    train_lasso_regression,
    train_linear_regression,
    train_neural_network_regressor,
    train_random_forest_regressor,
    train_ridge_regression,
    train_stacking_ensemble,
    train_svm_regressor,
    train_xgboost_regressor,
)

ZONE_COLS = [
    "geographic_zone_Asia Pacific",
    "geographic_zone_Europe",
    "geographic_zone_Northern America",
    "geographic_zone_Africa",
    "geographic_zone_Latin America",
    "geographic_zone_Middle East",
]

def _split_data(df: pd.DataFrame, target_col: str, leak_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    data = df.dropna(subset=[target_col]).copy()
    drop_cols = [target_col] + [c for c in leak_cols if c in data.columns]
    X = data.drop(columns=drop_cols)
    y = data[target_col]

    X = X.select_dtypes(include=[np.number, "bool"]).astype(float)
    data["zone"] = data[ZONE_COLS].idxmax(axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=data["zone"],
        random_state=42,
    )
    return X_train, X_test, y_train, y_test, X.columns.tolist()

def _train_baseline_models(X_train, X_test, y_train, y_test):
    train_functions = [
        train_linear_regression,
        train_ridge_regression,
        train_lasso_regression,
        train_decision_tree_regressor,
        train_random_forest_regressor,
        train_knn_regressor,
        train_svm_regressor,
        train_neural_network_regressor,
        train_xgboost_regressor,
        train_stacking_ensemble,
    ]

    results = []
    fitted = {}
    for fn in train_functions:
        result, model = fn(X_train, X_test, y_train, y_test)
        results.append(result)
        fitted[result.model] = model
    return results, fitted

# Select best model based on test R² with tie-breaking on RMSE, MAE, and overfitting gap
def _select_best_model(results_df: pd.DataFrame):
    metrics = results_df.copy()
    metrics["source"] = "baseline"
    metrics["overfit_gap"] = (metrics["train_r2"] - metrics["test_r2"]).abs()

    eligible = metrics[metrics["test_r2"] >= 0.75].copy()
    if eligible.empty:
        eligible = metrics.copy()

    eligible = eligible.sort_values(["test_rmse", "test_mae", "overfit_gap", "test_r2"], ascending=[True, True, True, False]).reset_index(drop=True)
    return eligible.iloc[0], metrics

def _update_registry(
    models_dir: Path,
    registry_key: str,
    model_file: str,
    feature_file: str,
    metrics_csv: str,
    metrics_json: str,
    best_row: pd.Series,
    target_col: str,
    best_model_name: str,
    feature_count: int,
):
    registry_path = models_dir / "model_registry.json"
    registry: Dict = {}
    if registry_path.exists():
        with open(registry_path, "r", encoding="utf-8") as f:
            registry = json.load(f)

    if "models" not in registry:
        registry["models"] = {}

    registry["models"][registry_key] = {
        "model_type": model_file,
        "features_file": feature_file,
        "metrics_files": {
            "all_models_csv": metrics_csv,
            "all_models_json": metrics_json,
        },
        "selection_criteria": {
            "min_test_r2": 0.75,
            "priority_order": ["test_rmse", "test_mae", "overfit_gap"],
        },
        "metrics": {
            "test_rmse": float(best_row["test_rmse"]),
            "test_mae": float(best_row["test_mae"]),
            "test_r2": float(best_row["test_r2"]),
            "train_r2": float(best_row["train_r2"]),
            "train_rmse": float(best_row["train_rmse"]),
            "train_mae": float(best_row["train_mae"]),
            "overfit_gap": float(best_row["overfit_gap"]),
            "target_transform": "log1p",
            "target_column": target_col,
        },
        "model_name": best_model_name,
        "source": str(best_row["source"]),
        "feature_count": feature_count,
        "training_date": pd.Timestamp.now().isoformat(),
    }

    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


def run_training_pipeline(
    feature_path: Path,
    models_dir: Path,
    *,
    target_col: str,
    leak_cols: List[str],
    registry_key: str,
    model_file_name: str,
    feature_file_name: str,
    metrics_csv_name: str,
    metrics_json_name: str,
) -> Dict:
    df = pd.read_csv(feature_path)
    X_train, X_test, y_train, y_test, feature_cols = _split_data(df, target_col, leak_cols)

    results, fitted_models = _train_baseline_models(X_train, X_test, y_train, y_test)
    results_df = metrics_to_frame(results)
    best_row, all_metrics = _select_best_model(results_df)
    best_model_name = str(best_row["model"])
    best_model = fitted_models[best_model_name]

    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / model_file_name
    joblib.dump(best_model, model_path)

    feature_path_txt = models_dir / feature_file_name
    with open(feature_path_txt, "w", encoding="utf-8") as f:
        f.write(f"# Features used for {registry_key}\n")
        for i, col in enumerate(feature_cols, start=1):
            f.write(f"{i}. {col}\n")

    metrics_csv_path = models_dir / metrics_csv_name
    metrics_json_path = models_dir / metrics_json_name
    all_metrics.sort_values(["test_rmse", "test_mae"]).to_csv(metrics_csv_path, index=False)
    all_metrics.sort_values(["test_rmse", "test_mae"]).to_json(metrics_json_path, orient="records", indent=2)

    _update_registry(
        models_dir=models_dir,
        registry_key=registry_key,
        model_file=model_file_name,
        feature_file=feature_file_name,
        metrics_csv=metrics_csv_name,
        metrics_json=metrics_json_name,
        best_row=best_row,
        target_col=target_col,
        best_model_name=best_model_name,
        feature_count=len(feature_cols),
    )

    return {
        "best_model": best_model_name,
        "model_path": str(model_path),
        "metrics_csv": str(metrics_csv_path),
        "metrics_json": str(metrics_json_path),
        "feature_count": len(feature_cols),
    }

def run_price_training_pipeline(feature_path: Path, models_dir: Path) -> Dict:
    return run_training_pipeline(
        feature_path=feature_path,
        models_dir=models_dir,
        target_col="ttm_avg_rate",
        leak_cols=["ttm_revenue"],
        registry_key="best_ttm_avg_rate",
        model_file_name="best_ttm_avg_rate_model.pkl",
        feature_file_name="best_ttm_avg_rate_features.txt",
        metrics_csv_name="all_price_model_metrics.csv",
        metrics_json_name="all_price_model_metrics.json",
    )

def run_revenue_training_pipeline(feature_path: Path, models_dir: Path) -> Dict:
    return run_training_pipeline(
        feature_path=feature_path,
        models_dir=models_dir,
        target_col="ttm_revenue",
        leak_cols=["ttm_avg_rate"],
        registry_key="best_ttm_revenue",
        model_file_name="best_ttm_revenue_model.pkl",
        feature_file_name="best_ttm_revenue_features.txt",
        metrics_csv_name="all_revenue_model_metrics.csv",
        metrics_json_name="all_revenue_model_metrics.json",
    )
