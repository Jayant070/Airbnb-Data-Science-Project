from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


@dataclass
class ModelResult:
    model: str
    train_r2: float
    test_r2: float
    train_rmse: float
    test_rmse: float
    train_mae: float
    test_mae: float


def _build_regression_pipeline(estimator: BaseEstimator, scale_features: bool) -> Pipeline:
    steps = []
    if scale_features:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def _evaluate_fitted_model(
    model_name: str,
    fitted_model,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[ModelResult, object]:
    train_predictions = fitted_model.predict(x_train)
    test_predictions = fitted_model.predict(x_test)

    metrics = ModelResult(
        model=model_name,
        train_r2=r2_score(y_train, train_predictions),
        test_r2=r2_score(y_test, test_predictions),
        train_rmse=float(np.sqrt(mean_squared_error(y_train, train_predictions))),
        test_rmse=float(np.sqrt(mean_squared_error(y_test, test_predictions))),
        train_mae=float(mean_absolute_error(y_train, train_predictions)),
        test_mae=float(mean_absolute_error(y_test, test_predictions)),
    )
    return metrics, fitted_model


def _perform_hyperparameter_search(
    base_estimator: BaseEstimator,
    param_grid: dict,
    x_train: pd.DataFrame,
    y_train,
    search_type: str = "random",
    n_iter: int = 20,
    cv: int = 3,
) -> BaseEstimator:
    """Perform hyperparameter search using GridSearchCV or RandomizedSearchCV.
    
    Args:
        base_estimator: Base estimator to tune
        param_grid: Dictionary of parameters to search
        x_train: Training features
        y_train: Training target
        search_type: 'random' for RandomizedSearchCV, 'grid' for GridSearchCV
        n_iter: Number of iterations for random search
        cv: Number of cross-validation folds
    
    Returns:
        Best estimator found during search
    """
    if search_type == "random":
        search = RandomizedSearchCV(
            base_estimator,
            param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            random_state=42,
            verbose=0,
        )
    else:
        search = GridSearchCV(
            base_estimator,
            param_grid,
            cv=cv,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            verbose=0,
        )
    
    search.fit(x_train, y_train)
    return search.best_estimator_


# Hyperparameter grids for top algorithms
_HYPERPARAMETER_GRIDS = {
    "random_forest": {
        "n_estimators": [120, 180, 240],
        "max_depth": [8, 12, 16],
        "min_samples_leaf": [2, 4, 8],
        "min_samples_split": [5, 10],
        "max_features": ["sqrt", "log2"],
    },
    "xgboost": {
        "n_estimators": [120, 180, 240],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.01, 0.03, 0.05],
        "subsample": [0.7, 0.8],
        "colsample_bytree": [0.7, 0.8],
        "min_child_weight": [3, 5],
        "gamma": [0.0, 0.1],
        "reg_alpha": [0.1, 0.5],
        "reg_lambda": [1.0, 1.5, 2.0],
    },
    "knn": {
        "n_neighbors": [5, 10, 15, 20, 25],
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    },
    "svm": {
        "C": [0.1, 1.0, 10.0, 100.0],
        "epsilon": [0.01, 0.05, 0.1, 0.2],
        "loss": ["epsilon_insensitive", "squared_epsilon_insensitive"],
    },
    "decision_tree": {
        "max_depth": [8, 10, 12, 15, 20],
        "min_samples_leaf": [2, 5, 10, 15],
        "min_samples_split": [2, 5, 10],
        "max_features": ["sqrt", "log2"],
    },
    "lasso": {
        "alpha": [0.0001, 0.0005, 0.001, 0.005, 0.01],
    },
    "ridge": {
        "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
    },
}


def train_price_model(
    model_name: str,
    estimator: BaseEstimator,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    *,
    scale_features: bool = False,
    use_hyperparameter_search: bool = False,
    param_grid: dict | None = None,
    search_type: str = "random",
    n_iter_search: int = 20,
    cv_folds: int = 3,
) -> tuple[ModelResult, object]:
    """Train a price prediction model with optional hyperparameter search.
    
    Args:
        model_name: Name of the model
        estimator: Base scikit-learn estimator
        x_train, x_test: Training and test features
        y_train, y_test: Training and test targets
        scale_features: Whether to add StandardScaler to the pipeline
        use_hyperparameter_search: Whether to perform hyperparameter tuning
        param_grid: Hyperparameter grid (required if use_hyperparameter_search=True)
        search_type: 'random' or 'grid' search
        n_iter_search: Number of random search iterations
        cv_folds: Number of cross-validation folds
    
    Returns:
        Tuple of (ModelResult metrics, fitted model pipeline)
    """
    # Perform hyperparameter search if enabled
    if use_hyperparameter_search and param_grid:
        y_train_array = y_train.values if isinstance(y_train, pd.Series) else y_train
        estimator = _perform_hyperparameter_search(
            estimator,
            param_grid,
            x_train,
            y_train_array,
            search_type=search_type,
            n_iter=n_iter_search,
            cv=cv_folds,
        )

    # Target is already transformed upstream if needed; do not log here again.
    fitted_model = _build_regression_pipeline(estimator, scale_features=scale_features)
    fitted_model.fit(x_train, y_train)
    return _evaluate_fitted_model(model_name, fitted_model, x_train, x_test, y_train, y_test)


def train_linear_regression(x_train, x_test, y_train, y_test, use_hyperparameter_search: bool = False):
    return train_price_model(
        "Linear Regression",
        LinearRegression(),
        x_train,
        x_test,
        y_train,
        y_test,
        scale_features=True,
    )


def train_ridge_regression(
    x_train, x_test, y_train, y_test, alpha: float = 1.0, use_hyperparameter_search: bool = False
):
    param_grid = _HYPERPARAMETER_GRIDS["ridge"] if use_hyperparameter_search else None
    return train_price_model(
        "Ridge Regression",
        Ridge(alpha=alpha, random_state=42),
        x_train,
        x_test,
        y_train,
        y_test,
        scale_features=True,
        use_hyperparameter_search=use_hyperparameter_search,
        param_grid=param_grid,
        search_type="grid",
    )


def train_lasso_regression(
    x_train, x_test, y_train, y_test, alpha: float = 0.0005, use_hyperparameter_search: bool = False
):
    param_grid = _HYPERPARAMETER_GRIDS["lasso"] if use_hyperparameter_search else None
    return train_price_model(
        "Lasso Regression",
        Lasso(alpha=alpha, max_iter=10000, random_state=42),
        x_train,
        x_test,
        y_train,
        y_test,
        scale_features=True,
        use_hyperparameter_search=use_hyperparameter_search,
        param_grid=param_grid,
        search_type="grid",
    )


def train_decision_tree_regressor(
    x_train, x_test, y_train, y_test, use_hyperparameter_search: bool = False
):
    param_grid = _HYPERPARAMETER_GRIDS["decision_tree"] if use_hyperparameter_search else None
    return train_price_model(
        "Decision Tree Regressor",
        DecisionTreeRegressor(max_depth=12, min_samples_leaf=10, random_state=42),
        x_train,
        x_test,
        y_train,
        y_test,
        scale_features=False,
        use_hyperparameter_search=use_hyperparameter_search,
        param_grid=param_grid,
        search_type="random",
        n_iter_search=15,
    )


def train_random_forest_regressor(
    x_train,
    x_test,
    y_train,
    y_test,
    use_hyperparameter_search: bool = False,
    *,
    n_iter_search: int = 12,
    cv_folds: int = 3,
):
    param_grid = _HYPERPARAMETER_GRIDS["random_forest"] if use_hyperparameter_search else None
    return train_price_model(
        "Random Forest Regressor",
        RandomForestRegressor(
            n_estimators=180,
            max_depth=16,
            min_samples_leaf=4,
            min_samples_split=10,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42,
        ),
        x_train,
        x_test,
        y_train,
        y_test,
        scale_features=False,
        use_hyperparameter_search=use_hyperparameter_search,
        param_grid=param_grid,
        search_type="random",
        n_iter_search=n_iter_search,
        cv_folds=cv_folds,
    )


def train_xgboost_regressor(
    x_train,
    x_test,
    y_train,
    y_test,
    use_hyperparameter_search: bool = False,
    *,
    n_iter_search: int = 12,
    cv_folds: int = 3,
):
    param_grid = _HYPERPARAMETER_GRIDS["xgboost"] if use_hyperparameter_search else None

    xgb_kwargs = {
        "n_estimators": 240,
        "learning_rate": 0.03,
        "max_depth": 4,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "min_child_weight": 5,
        "gamma": 0.1,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "objective": "reg:squarederror",
        "random_state": 42,
        "n_jobs": -1,
    }

    return train_price_model(
        "XGBoost Regressor",
        XGBRegressor(**xgb_kwargs),
        x_train,
        x_test,
        y_train,
        y_test,
        scale_features=False,
        use_hyperparameter_search=use_hyperparameter_search,
        param_grid=param_grid,
        search_type="random",
        n_iter_search=n_iter_search,
        cv_folds=cv_folds,
    )


def train_knn_regressor(
    x_train, x_test, y_train, y_test, use_hyperparameter_search: bool = False
):
    param_grid = _HYPERPARAMETER_GRIDS["knn"] if use_hyperparameter_search else None
    return train_price_model(
        "KNN Regressor",
        KNeighborsRegressor(n_neighbors=15, weights="distance", n_jobs=-1),
        x_train,
        x_test,
        y_train,
        y_test,
        scale_features=True,
        use_hyperparameter_search=use_hyperparameter_search,
        param_grid=param_grid,
        search_type="grid",
    )


def train_svm_regressor(
    x_train, x_test, y_train, y_test, use_hyperparameter_search: bool = False
):
    param_grid = _HYPERPARAMETER_GRIDS["svm"] if use_hyperparameter_search else None
    return train_price_model(
        "Support Vector Regressor",
        LinearSVR(C=1.0, epsilon=0.1, max_iter=10000, random_state=42),
        x_train,
        x_test,
        y_train,
        y_test,
        scale_features=True,
        use_hyperparameter_search=use_hyperparameter_search,
        param_grid=param_grid,
        search_type="random",
        n_iter_search=20,
    )


def train_neural_network_regressor(x_train, x_test, y_train, y_test):
    return train_price_model(
        "Neural Network Regressor",
        MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=0.0001,
            batch_size=256,
            learning_rate_init=0.001,
            max_iter=200,
            early_stopping=True,
            n_iter_no_change=15,
            random_state=42,
        ),
        x_train,
        x_test,
        y_train,
        y_test,
        scale_features=True,
    )


def train_stacking_ensemble(
    x_train,
    x_test,
    y_train,
    y_test,
    use_hyperparameter_search: bool = False,
    *,
    cv_folds: int = 3,
    deep_search: bool = False,
):
    base_estimators = [
        (
            "ridge",
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0, random_state=42)),
            ]),
        ),
        (
            "rf",
            RandomForestRegressor(
                n_estimators=120,
                max_depth=None,
                min_samples_leaf=2,
                n_jobs=1,
                random_state=42,
            ),
        ),
        (
            "xgb",
            XGBRegressor(
                n_estimators=120,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=1,
            ),
        ),
    ]

    stacking_model = StackingRegressor(
        estimators=base_estimators,
        final_estimator=Ridge(alpha=1.0, random_state=42),
        passthrough=False,
        n_jobs=-1,
    )

    # Note: Stacking hyperparameter search is limited - mainly tune final estimator
    param_grid = None
    if use_hyperparameter_search:
        alpha_grid = [0.1, 1.0, 10.0]
        if deep_search:
            alpha_grid = [0.01, 0.05, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
        param_grid = {"final_estimator__alpha": alpha_grid}

    return train_price_model(
        "Stacking Ensemble",
        stacking_model,
        x_train,
        x_test,
        y_train,
        y_test,
        scale_features=False,
        use_hyperparameter_search=use_hyperparameter_search,
        param_grid=param_grid,
        search_type="grid",
        cv_folds=cv_folds,
    )


def metrics_to_frame(results: Iterable[ModelResult]) -> pd.DataFrame:
    return pd.DataFrame([result.__dict__ for result in results]).sort_values("test_rmse").reset_index(drop=True)


def plot_predicted_vs_actual(
    y_true,
    y_pred,
    model_name: str,
    *,
    ax=None,
):
    """Plot predicted price against actual price on the original scale."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    min_value = float(min(y_true_array.min(), y_pred_array.min()))
    max_value = float(max(y_true_array.max(), y_pred_array.max()))

    ax.scatter(y_true_array, y_pred_array, alpha=0.25, s=12, color="royalblue")
    ax.plot([min_value, max_value], [min_value, max_value], "r--", lw=2)
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")
    ax.set_title(f"{model_name}: Predicted vs Actual")
    ax.grid(alpha=0.3)
    return ax
