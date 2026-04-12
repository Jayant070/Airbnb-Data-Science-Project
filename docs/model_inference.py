"""
Local model inference helper for current production artifacts.

Usage:
    from model_inference import ModelLoader

    loader = ModelLoader("../models")

    price_model = loader.load_model("best_ttm_avg_rate")
    pred = price_model.predict(df)

    revenue_model = loader.load_model("best_ttm_revenue")
    pred = revenue_model.predict(df)
"""

from pathlib import Path
import json
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd


class ModelWrapper:
    def __init__(self, model, features: List[str], target_name: str, target_transform: str = "none"):
        self.model = model
        self.features = features
        self.target_name = target_name
        self.target_transform = target_transform

    def predict(self, X: pd.DataFrame, return_business_scale: bool = True) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        missing = [col for col in self.features if col not in X.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        raw_pred = self.model.predict(X[self.features])

        if return_business_scale and self.target_transform == "log1p":
            return np.maximum(np.expm1(raw_pred), 0.0)

        return raw_pred


class ModelLoader:
    def __init__(self, models_dir: str):
        self.models_dir = Path(models_dir)
        with open(self.models_dir / "model_registry.json", "r", encoding="utf-8") as f:
            self.registry = json.load(f)

    @staticmethod
    def _load_features_from_txt(path: Path) -> List[str]:
        features: List[str] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text or text.startswith("#"):
                    continue
                if ". " in text:
                    _, feature = text.split(". ", 1)
                    features.append(feature.strip())
                else:
                    features.append(text)
        if not features:
            raise ValueError(f"No features found in {path}")
        return features

    def _resolve_model_file(self, model_key: str, model_cfg: Dict) -> Path:
        direct = model_cfg.get("model_file")
        if direct:
            candidate = self.models_dir / direct
            if candidate.exists():
                return candidate

        # Registry currently stores revenue model field as model_type.
        fallback = model_cfg.get("model_type")
        if fallback:
            candidate = self.models_dir / fallback
            if candidate.exists():
                return candidate

        candidate = self.models_dir / f"{model_key}_model.pkl"
        if candidate.exists():
            return candidate

        raise FileNotFoundError(f"Could not resolve model file for key: {model_key}")

    def load_model(self, model_key: str):
        models = self.registry.get("models", {})
        if model_key not in models:
            raise ValueError(f"Unknown model key: {model_key}. Available: {list(models.keys())}")

        cfg = models[model_key]
        model_path = self._resolve_model_file(model_key, cfg)

        features_file = cfg.get("features_file")
        if not features_file:
            raise ValueError(f"Missing features_file in registry for model key: {model_key}")
        features_path = self.models_dir / features_file
        if not features_path.exists():
            raise FileNotFoundError(f"Features file not found: {features_path}")

        metrics = cfg.get("metrics", {})
        target_name = metrics.get("target_column", model_key)
        target_transform = metrics.get("target_transform", "none")

        model = joblib.load(model_path)
        features = self._load_features_from_txt(features_path)

        return ModelWrapper(
            model=model,
            features=features,
            target_name=target_name,
            target_transform=target_transform,
        )


if __name__ == "__main__":
    loader = ModelLoader("../models")
    _ = list(loader.registry.get("models", {}).keys())
