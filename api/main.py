"""
FastAPI Application for Airbnb Models Serving
Production-ready API with model caching, validation, and error handling
"""

import os
import json
import logging
import sys
import re
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import traceback

# Data & ML Libraries
import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans

# FastAPI & Web Framework
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Add scripts directory to path for importing geo_encoding
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "geoencoding"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "data" / "main"))
from geo_encoding import resolve_city_distance
from amenties_utility import create_amenity_features
from listing_type_utils import categorize_listing

# Configuration

# Setup Paths
project_root = Path(__file__).parent.parent
models_dir = project_root / "models"
docs_dir = project_root / "docs"
scripts_dir = project_root / "scripts"
feature_data_path = project_root / "data" / "processed" / "main" / "feature_engineered.csv"

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Debug switch for printing final model input features and values.
DEBUG_FEATURE_LOGGING = os.getenv("DEBUG_FEATURE_LOGGING", "true").strip().lower() in {"1", "true", "yes", "y", "on"}

CANCELLATION_ORDER = {
    "Full Refundable Until Check-in": 0,
    "Full Refundable Until 24 Hours Before Check-in": 1,
    "Full Refundable Until 72 Hours Before Check-in": 2,
    "Refundable": 3,
    "Flexible": 4,
    "Moderate": 5,
    "Limited": 6,
    "Firm": 7,
    "Strict": 8,
    "Non-refundable": 9,
    "Super Strict 30 Days": 10,
    "Super Strict 60 Days": 11,
}

# Training-time feature engineering constants.
# These mirror the values used in Notebooks/04_Training.ipynb so inference sees
# the same feature math the model was trained on.
TRAINING_NUM_REVIEWS_MAX = 399.0
TRAINING_DISTANCE_FROM_CITY_CENTER_MAX = 16.0

# FastAPI app setup

app = FastAPI(
    title="Airbnb Prediction API",
    description="Production-ready API for serving trained Airbnb prediction models",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info(f"FastAPI app initialized. Models dir: {models_dir}")

# Pydantic models

class ListingInput(BaseModel):
    bedrooms: int = Field(..., ge=0, le=50)
    beds: int = Field(..., ge=0, le=50)
    baths: int = Field(..., ge=0, le=50)
    guests: int = Field(..., ge=0, le=100)
    amenities_count: Optional[int] = Field(None, ge=0)
    photos_count: int = Field(..., ge=0)
    superhost: int = Field(..., ge=0, le=1)
    num_reviews: int = Field(..., ge=0)
    avg_rating: float = Field(..., ge=1, le=5)
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    distance_from_city_center: Optional[float] = Field(None, ge=0)
    cancellation_policy: Union[str, float, int] = Field(..., description="Cancellation policy label or encoded numeric value")
    min_nights: int = Field(..., ge=0, le=1000)
    cleaning_fee: float = Field(..., ge=0, le=10000)
    extra_guest_fee: float = Field(..., ge=0, le=1000)
    registration: int = Field(..., ge=0, le=1)
    professional_management: int = Field(..., ge=0, le=1)
    listing_type: str = Field("other", description="Listing type (mapped to training categories)")
    room_type: str = Field("private_room", description="Room type: hotel_room, private_room, shared_room")
    city_name: Optional[str] = Field(None, description="Deprecated for inference; coordinates are used for georesolution")
    city_population: Optional[float] = Field(None, ge=0, description="Optional city population; auto-resolved from geoencoding when available")
    amenities: Optional[Union[str, List[str]]] = Field(
        None,
        description="Amenities as comma-separated string or list of strings",
    )

    # Optional enriched features used directly by trained models.
    total_amenities: Optional[float] = Field(None, ge=0)
    luxury_score: Optional[float] = Field(None, ge=0)
    rarity_score: Optional[float] = Field(None, ge=0)
    basic_count: Optional[float] = Field(None, ge=0)
    comfort_count: Optional[float] = Field(None, ge=0)
    kitchen_count: Optional[float] = Field(None, ge=0)
    safety_count: Optional[float] = Field(None, ge=0)
    outdoor_count: Optional[float] = Field(None, ge=0)
    family_count: Optional[float] = Field(None, ge=0)
    services_count: Optional[float] = Field(None, ge=0)
    comfort_to_total_ratio: Optional[float] = Field(None, ge=0)
    has_pool: Optional[int] = Field(0, ge=0, le=1)
    has_hot_tub: Optional[int] = Field(0, ge=0, le=1)
    has_gym: Optional[int] = Field(0, ge=0, le=1)
    has_beach_access: Optional[int] = Field(0, ge=0, le=1)
    has_dedicated_workspace: Optional[int] = Field(0, ge=0, le=1)
    has_pets_allowed: Optional[int] = Field(0, ge=0, le=1)
    has_free_parking_on_premises: Optional[int] = Field(0, ge=0, le=1)
    has_air_conditioning: Optional[int] = Field(0, ge=0, le=1)
    nearby_listing_density: Optional[float] = Field(0.0, ge=0)
    ttm_blocked_days: Optional[float] = Field(0.0, ge=0)
    ttm_total_days: Optional[float] = Field(365.0, ge=1)


class PredictionResponse(BaseModel):
    success: bool
    model: str
    target: str
    prediction: float
    prediction_formatted: str
    timestamp: str
    request_id: Optional[str] = None


class HealthCheckResponse(BaseModel):
    status: str
    models_loaded: Dict[str, bool]
    models_directory: str
    timestamp: str


class BatchListingInput(BaseModel):
    listings: List[ListingInput]
    model: str


class BatchPredictionResponse(BaseModel):
    success: bool
    model: str
    total_listings: int
    successful_predictions: int
    failed_predictions: int
    predictions: List[Dict[str, Any]]
    errors: List[Dict[str, Any]]
    processing_time_seconds: float
    timestamp: str


class CompetitiveAnalysisResponse(BaseModel):
    success: bool
    cluster_id: int
    cluster_size: int
    market_position: str
    competitiveness_score: float
    price_vs_cluster: float
    predicted_price: float
    cluster_median_price: float
    percentile_ranks: Dict[str, float]
    timestamp: str


# Model cache

class ModelCache:
    def __init__(self, models_dir):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.resolved_model_files = {}
        self.feature_lists = {}
        self.model_registry = None

    def _load_features_from_txt(self, file_path: Path) -> List[str]:
        if not file_path.exists():
            raise FileNotFoundError(f"Feature file not found: {file_path}")

        features: List[str] = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                # Format: "1. feature_name"
                if '. ' in line:
                    _, feature = line.split('. ', 1)
                    features.append(feature.strip())
        if not features:
            raise ValueError(f"No features found in {file_path}")
        return features

    def load_feature_lists(self):
        if self.feature_lists:
            return self.feature_lists

        mapping = {
            'model1_price': self.models_dir / 'model1_price_features.txt',
            'model2_revenue': self.models_dir / 'model2_revenue_features.txt',
            'best_ttm_avg_rate': self.models_dir / 'best_ttm_avg_rate_features.txt',
            'best_ttm_revenue': self.models_dir / 'best_ttm_revenue_features.txt',
        }
        for model_key, path in mapping.items():
            if path.exists():
                self.feature_lists[model_key] = self._load_features_from_txt(path)

        logger.info(f"Loaded feature lists: {list(self.feature_lists.keys())}")
        return self.feature_lists

    def load_model_registry(self):
        if self.model_registry is None:
            registry_path = self.models_dir / 'model_registry.json'
            if registry_path.exists():
                with open(registry_path, 'r') as f:
                    self.model_registry = json.load(f)
            else:
                self.model_registry = {}
        return self.model_registry

    def _resolve_model_path(self, model_name: str) -> Path:
        """Resolve a requested model key to an existing model artifact path."""
        candidate_files: List[str] = [f"{model_name}.pkl"]

        # Backward-compatible aliases for renamed artifacts.
        if model_name == 'model1_xgb_price':
            registry = self.load_model_registry()
            models = registry.get('models', {}) if isinstance(registry, dict) else {}
            registry_file = models.get('model1_price', {}).get('model_file')
            if registry_file:
                candidate_files.insert(0, registry_file)
            candidate_files.append('model1_best_price.pkl')
        elif model_name == 'model2_xgb_revenue':
            registry = self.load_model_registry()
            models = registry.get('models', {}) if isinstance(registry, dict) else {}
            registry_file = models.get('model2_revenue', {}).get('model_file')
            if registry_file:
                candidate_files.insert(0, registry_file)
        elif model_name == 'best_ttm_avg_rate_model':
            candidate_files.insert(0, 'best_ttm_avg_rate_model.pkl')
        elif model_name == 'best_ttm_revenue_model':
            candidate_files.insert(0, 'best_ttm_revenue_model.pkl')

        seen = set()
        for file_name in candidate_files:
            if not file_name or file_name in seen:
                continue
            seen.add(file_name)
            model_path = self.models_dir / file_name
            if model_path.exists():
                return model_path

        raise FileNotFoundError(
            f"Model file not found for '{model_name}'. Tried: {candidate_files}"
        )
    
    def load_model(self, model_name: str):
        if model_name not in self.models:
            model_file = self._resolve_model_path(model_name)
            self.models[model_name] = joblib.load(model_file)
            self.resolved_model_files[model_name] = model_file.name
            logger.info(f"Loaded model: {model_name} from {model_file.name}")
        return self.models[model_name]

    def get_loaded_model_filename(self, model_name: str) -> Optional[str]:
        return self.resolved_model_files.get(model_name)
    
    def get_features(self, model_key: str):
        feature_lists = self.load_feature_lists()
        if model_key in feature_lists:
            return feature_lists[model_key]
        raise ValueError(f"Features not found for model: {model_key}")

    def get_target_transform(self, model_key: str, model_name: Optional[str] = None) -> str:
        # Some legacy artifacts already include inverse transform in-model.
        if model_name:
            loaded_file = self.get_loaded_model_filename(model_name)
            if loaded_file in {'model1_best_price.pkl'}:
                return 'none'

        registry = self.load_model_registry()
        models = registry.get('models', {}) if isinstance(registry, dict) else {}
        if model_key == 'model1_price':
            model_cfg = models.get('model1_price', {})
            return model_cfg.get('target_transform') or model_cfg.get('metrics', {}).get('target_transform', 'log1p')
        if model_key == 'model2_revenue':
            model_cfg = models.get('model2_revenue', {})
            return model_cfg.get('target_transform') or model_cfg.get('metrics', {}).get('target_transform', 'log1p')
        if model_key == 'best_ttm_avg_rate':
            model_cfg = models.get('best_ttm_avg_rate', {})
            return model_cfg.get('target_transform') or model_cfg.get('metrics', {}).get('target_transform', 'log1p')
        if model_key == 'best_ttm_revenue':
            model_cfg = models.get('best_ttm_revenue', {})
            return model_cfg.get('target_transform') or model_cfg.get('metrics', {}).get('target_transform', 'log1p')
        return 'none'
    
    def verify_all_models_loaded(self):
        return {
            'model1_xgb_price': (
                (self.models_dir / 'model1_xgb_price.pkl').exists()
                or (self.models_dir / 'model1_best_price.pkl').exists()
            ),
            'model2_xgb_revenue': (self.models_dir / 'model2_xgb_revenue.pkl').exists(),
            'model1_price_features': (self.models_dir / 'model1_price_features.txt').exists(),
            'model2_revenue_features': (self.models_dir / 'model2_revenue_features.txt').exists(),
            'model_registry': (self.models_dir / 'model_registry.json').exists(),
            'best_ttm_avg_rate_model': (self.models_dir / 'best_ttm_avg_rate_model.pkl').exists(),
            'best_ttm_avg_rate_features': (self.models_dir / 'best_ttm_avg_rate_features.txt').exists(),
            'best_ttm_revenue_model': (self.models_dir / 'best_ttm_revenue_model.pkl').exists(),
            'best_ttm_revenue_features': (self.models_dir / 'best_ttm_revenue_features.txt').exists(),
        }

cache = ModelCache(models_dir)

# Preprocessing functions

def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _normalize_listing_type(value: str) -> str:
    category = categorize_listing(value)
    if category in {"apartment", "townhouse", "unknown"}:
        return "other"
    return category if category else "other"


def _normalize_amenities_input(amenities: Optional[Union[str, List[str]]]) -> str:
    if amenities is None:
        return ""
    if isinstance(amenities, list):
        return ", ".join(str(a).strip() for a in amenities if str(a).strip())

    text = str(amenities)
    for sep in [";", "|", "\n", "\t", "/"]:
        text = text.replace(sep, ",")

    parts = [p.strip() for p in text.split(",") if p.strip()]
    return ", ".join(parts)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _encode_cancellation_policy(value: Union[str, float, int]) -> float:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(max(0.0, value))

    text = str(value or "").strip()
    if not text:
        return 4.0

    if text in CANCELLATION_ORDER:
        return float(CANCELLATION_ORDER[text])

    # Fallback: case-insensitive exact match
    lowered = text.lower()
    for key, encoded in CANCELLATION_ORDER.items():
        if key.lower() == lowered:
            return float(encoded)

    try:
        return float(max(0.0, float(text)))
    except (TypeError, ValueError):
        logger.warning(f"Unknown cancellation policy '{text}', defaulting to Flexible (4)")
        return 4.0


def _compute_amenities_count(amenities: Optional[Union[str, List[str]]], fallback_count: Optional[int]) -> int:
    if isinstance(amenities, list):
        cleaned = [str(a).strip() for a in amenities if str(a).strip()]
        if cleaned:
            return len(cleaned)

    if isinstance(amenities, str):
        normalized = _normalize_amenities_input(amenities)
        if normalized:
            return len([item for item in normalized.split(",") if item.strip()])

    if fallback_count is not None:
        return int(max(0, fallback_count))

    return 0


def _apply_training_feature_transforms(df: pd.DataFrame, expected_features: List[str]) -> pd.DataFrame:
    """Apply the same in-place feature transforms used during feature engineering."""
    log1p_feature_candidates = ["num_reviews", "distance_from_city_center"]
    for col in log1p_feature_candidates:
        if col in expected_features and col in df.columns:
            df[col] = np.log1p(np.clip(df[col], a_min=0.0, a_max=None))
    return df


@lru_cache(maxsize=1)
def _load_competitive_reference() -> tuple[pd.DataFrame, KMeans]:
    if not feature_data_path.exists():
        raise FileNotFoundError(f"Competitive reference data not found: {feature_data_path}")

    df = pd.read_csv(feature_data_path)
    required = ["latitude", "longitude", "ttm_avg_rate", "avg_rating", "num_reviews", "amenities_count"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column in reference data: {col}")

    ref = df[required].copy().dropna(subset=["latitude", "longitude", "ttm_avg_rate"])
    # Stored target is log-space in this project; convert back to business scale for comparisons.
    ref["ttm_avg_rate_business"] = np.maximum(np.expm1(ref["ttm_avg_rate"].astype(float)), 0.0)

    # Keep cluster count data-aware for smaller datasets.
    n_clusters = int(np.clip(len(ref) // 2000, 5, 20)) if len(ref) >= 1000 else 5
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    ref["cluster_id"] = model.fit_predict(ref[["latitude", "longitude"]])
    return ref, model


def engineer_features(listing: ListingInput, expected_features: List[str]) -> pd.DataFrame:
    """
    Comprehensive feature engineering pipeline:
    1. Start with raw input
    2. Add categorical encodings (one-hot encoding)
    3. Calculate derived features
    4. Determine geographic zone
    Returns DataFrame with all engineered features
    """
    listing_dict = listing.dict()
    latitude = float(listing_dict['latitude'])
    longitude = float(listing_dict['longitude'])
    cancellation_policy_encoded = _encode_cancellation_policy(listing_dict.get('cancellation_policy'))
    amenities_count = _compute_amenities_count(listing_dict.get('amenities'), listing_dict.get('amenities_count'))
    provided_distance = listing_dict.get('distance_from_city_center')
    provided_city_population = listing_dict.get('city_population')

    zone = 'Northern America'
    resolved_distance = 0.0
    resolved_city_population = 0.0

    try:
        # Resolve geo metadata from coordinates; explicit request values can override.
        geo_result = resolve_city_distance(
            cityname=None,
            latitude=latitude,
            longitude=longitude,
        )
        zone = geo_result.get('zone') or 'Northern America'
        resolved_distance = _safe_float(geo_result.get('distance_from_city_center', geo_result.get('distance_km')), 0.0)
        resolved_city_population = _safe_float(geo_result.get('city_population'), 0.0)
    except Exception as e:
        logger.warning(f"Failed to resolve city distance/zone via geoencoding: {e}")

    # API behavior: use direct values from request if provided, otherwise use geo-resolved values.
    distance_from_city_center = _safe_float(provided_distance, resolved_distance)
    city_population = _safe_float(provided_city_population, resolved_city_population)

    amenities_text = _normalize_amenities_input(listing_dict.get('amenities'))
    amenity_features: Dict[str, Any] = {}
    if amenities_text:
        try:
            amenity_df = create_amenity_features(pd.Series([amenities_text]))
            amenity_features = amenity_df.iloc[0].to_dict()
        except Exception as e:
            logger.warning(f"Failed to parse amenities using amenities utility: {e}")

    total_amenities = _safe_float(
        amenity_features.get('total_amenities')
        if 'total_amenities' in amenity_features
        else (
            listing_dict.get('total_amenities')
            if listing_dict.get('total_amenities') is not None
            else amenities_count
        )
    )

    # If amenity-category counts are not provided, derive deterministic proxies.
    basic_count = _safe_float(
        amenity_features.get('basic_count')
        if 'basic_count' in amenity_features
        else (listing_dict.get('basic_count') if listing_dict.get('basic_count') is not None else total_amenities * 0.20)
    )
    comfort_count = _safe_float(
        amenity_features.get('comfort_count')
        if 'comfort_count' in amenity_features
        else (listing_dict.get('comfort_count') if listing_dict.get('comfort_count') is not None else total_amenities * 0.20)
    )
    kitchen_count = _safe_float(
        amenity_features.get('kitchen_count')
        if 'kitchen_count' in amenity_features
        else (listing_dict.get('kitchen_count') if listing_dict.get('kitchen_count') is not None else total_amenities * 0.15)
    )
    safety_count = _safe_float(
        amenity_features.get('safety_count')
        if 'safety_count' in amenity_features
        else (listing_dict.get('safety_count') if listing_dict.get('safety_count') is not None else total_amenities * 0.15)
    )
    outdoor_count = _safe_float(
        amenity_features.get('outdoor_count')
        if 'outdoor_count' in amenity_features
        else (listing_dict.get('outdoor_count') if listing_dict.get('outdoor_count') is not None else total_amenities * 0.10)
    )
    family_count = _safe_float(
        amenity_features.get('family_count')
        if 'family_count' in amenity_features
        else (listing_dict.get('family_count') if listing_dict.get('family_count') is not None else total_amenities * 0.10)
    )
    services_count = _safe_float(
        amenity_features.get('services_count')
        if 'services_count' in amenity_features
        else (listing_dict.get('services_count') if listing_dict.get('services_count') is not None else total_amenities * 0.10)
    )

    comfort_to_total_ratio = listing_dict.get('comfort_to_total_ratio')
    if comfort_to_total_ratio is None:
        comfort_to_total_ratio = _safe_float(
            amenity_features.get('comfort_to_total_ratio', comfort_count / (total_amenities + 1e-6))
        )

    num_reviews = float(listing_dict['num_reviews'])
    avg_rating = float(listing_dict['avg_rating'])
    superhost = int(listing_dict['superhost'])
    ttm_blocked_days = float(listing_dict.get('ttm_blocked_days') or 0.0)
    ttm_total_days = float(listing_dict.get('ttm_total_days') or 365.0)

    feature_row = {
        'photos_count': float(listing_dict['photos_count']),
        'superhost': superhost,
        'latitude': latitude,
        'longitude': longitude,
        'guests': int(listing_dict['guests']),
        'bedrooms': int(listing_dict['bedrooms']),
        'beds': int(listing_dict['beds']),
        'baths': int(listing_dict['baths']),
        'registration': int(listing_dict['registration']),
        'professional_management': int(listing_dict['professional_management']),
        'min_nights': int(listing_dict['min_nights']),
        'cancellation_policy': cancellation_policy_encoded,
        'cleaning_fee': float(listing_dict['cleaning_fee']),
        'extra_guest_fee': float(listing_dict['extra_guest_fee']),
        'num_reviews': num_reviews,
        'ttm_blocked_days': ttm_blocked_days,
        'ttm_total_days': ttm_total_days,
        'avg_rating': avg_rating,
        'distance_from_city_center': distance_from_city_center,
        'city_population': city_population,
        'luxury_score': float(amenity_features.get('luxury_score', listing_dict.get('luxury_score') or 0.0)),
        'basic_count': basic_count,
        'comfort_count': comfort_count,
        'kitchen_count': kitchen_count,
        'safety_count': safety_count,
        'outdoor_count': outdoor_count,
        'family_count': family_count,
        'services_count': services_count,
        'has_pool': int(amenity_features.get('has_pool', listing_dict.get('has_pool') or 0)),
        'has_hot_tub': int(amenity_features.get('has_hot_tub', listing_dict.get('has_hot_tub') or 0)),
        'has_gym': int(amenity_features.get('has_gym', listing_dict.get('has_gym') or 0)),
        'has_beach_access': int(amenity_features.get('has_beach_access', listing_dict.get('has_beach_access') or 0)),
        'has_dedicated_workspace': int(amenity_features.get('has_dedicated_workspace', listing_dict.get('has_dedicated_workspace') or 0)),
        'has_pets_allowed': int(amenity_features.get('has_pets_allowed', listing_dict.get('has_pets_allowed') or 0)),
        'has_free_parking_on_premises': int(amenity_features.get('has_free_parking_on_premises', listing_dict.get('has_free_parking_on_premises') or 0)),
        'has_air_conditioning': int(amenity_features.get('has_air_conditioning', listing_dict.get('has_air_conditioning') or 0)),
        'total_amenities': total_amenities,
        'comfort_to_total_ratio': float(comfort_to_total_ratio),
        'rarity_score': float(amenity_features.get('rarity_score', listing_dict.get('rarity_score') or 0.0)),
        # Engineered features from training notebook
        'host_track_record': (superhost * 0.5) + ((avg_rating / 5.0) * 0.5),
        'review_frequency': num_reviews / (TRAINING_NUM_REVIEWS_MAX + 1.0),
        'is_city_center': int(distance_from_city_center < 2.0),
        'distance_zone_normalized': distance_from_city_center / (TRAINING_DISTANCE_FROM_CITY_CENTER_MAX + 0.1),
        'nearby_listing_density': float(listing_dict.get('nearby_listing_density') or 0.0),
        'review_density': num_reviews / (ttm_total_days + 1.0),
    }

    df = pd.DataFrame([feature_row])

    # Dynamic one-hot encoding aligned to model feature list.
    expected_listing_type_cols = [f for f in expected_features if f.startswith('listing_type_')]
    expected_room_type_cols = [f for f in expected_features if f.startswith('room_type_')]
    expected_zone_cols = [f for f in expected_features if f.startswith('geographic_zone_')]

    listing_type_value = _normalize_listing_type(listing_dict.get('listing_type', 'other'))
    room_type_value = _normalize_key(listing_dict.get('room_type', 'private_room'))
    zone_value = str(zone).strip()

    for col in expected_listing_type_cols:
        col_value = _normalize_key(col.replace('listing_type_', ''))
        df[col] = 1 if col_value == listing_type_value else 0

    # Fallback to listing_type_other if available and no direct match.
    if expected_listing_type_cols and int(df[expected_listing_type_cols].to_numpy().sum()) == 0 and 'listing_type_other' in df.columns:
        df['listing_type_other'] = 1

    for col in expected_room_type_cols:
        col_value = _normalize_key(col.replace('room_type_', ''))
        df[col] = 1 if col_value == room_type_value else 0

    for col in expected_zone_cols:
        zone_label = col.replace('geographic_zone_', '').strip()
        df[col] = 1 if zone_label == zone_value else 0

    if expected_zone_cols and int(df[expected_zone_cols].to_numpy().sum()) == 0 and 'geographic_zone_Northern America' in df.columns:
        df['geographic_zone_Northern America'] = 1

    # Ensure all expected features are present and ordered.
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0.0

    df = pd.DataFrame(df.loc[:, expected_features])
    return _apply_training_feature_transforms(df, expected_features)


def log_model_input_features(model_key: str, X: pd.DataFrame, request_id: str, listing_index: Optional[int] = None) -> None:
    """Print and log final engineered feature values sent to the model."""
    if not DEBUG_FEATURE_LOGGING:
        return

    if X.empty:
        logger.warning(f"[{request_id}] Empty feature frame for {model_key}")
        return

    suffix = f" listing_index={listing_index}" if listing_index is not None else ""
    feature_map = {col: X.iloc[0][col] for col in X.columns}

    print("\n" + "=" * 80)
    print(f"MODEL INPUT FEATURES | model={model_key} request_id={request_id}{suffix}")
    print("=" * 80)
    for k, v in feature_map.items():
        print(f"{k}: {v}")
    print("=" * 80 + "\n")

    logger.info(f"[{request_id}] Final model input features for {model_key}{suffix}: {json.dumps(feature_map, default=str)}")


def prepare_single_prediction(listing: ListingInput, features: List[str]) -> pd.DataFrame:
    """Prepare features for prediction with full feature engineering"""
    df = engineer_features(listing, features)
    
    nan_cols = df.columns[df.isna().any()].tolist()
    if nan_cols:
        raise ValueError(f"NaN values found in features: {nan_cols}")
    
    return df


# Exception handlers

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    logger.error(f"ValueError: {str(exc)}")
    return JSONResponse(
        status_code=422,
        content={"success": False, "error": f"Invalid input: {str(exc)}", "timestamp": datetime.now().isoformat()}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Error: {str(exc)}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": "Internal server error", "timestamp": datetime.now().isoformat()}
    )


# API endpoints

@app.get("/api/health", response_model=HealthCheckResponse)
async def health_check():
    try:
        models_status = cache.verify_all_models_loaded()
        all_ready = all(models_status.values())
        
        return HealthCheckResponse(
            status="healthy" if all_ready else "degraded",
            models_loaded=models_status,
            models_directory=str(models_dir),
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthCheckResponse(
            status="unhealthy",
            models_loaded={},
            models_directory=str(models_dir),
            timestamp=datetime.now().isoformat()
        )


@app.post("/api/predict/price", response_model=PredictionResponse)
async def predict_price(listing: ListingInput) -> PredictionResponse:
    try:
        request_id = f"price_{datetime.now().timestamp()}"
        model_name = "best_ttm_avg_rate_model"
        model = cache.load_model(model_name)
        features = cache.get_features("best_ttm_avg_rate")
        target_transform = cache.get_target_transform("best_ttm_avg_rate", model_name=model_name)

        X = prepare_single_prediction(listing, features)
        log_model_input_features("best_ttm_avg_rate", X, request_id)
        raw_prediction = float(model.predict(X)[0])
        prediction = float(np.maximum(np.expm1(raw_prediction), 0.0)) if target_transform == 'log1p' else raw_prediction

        logger.info(f"Price prediction: ${prediction:.2f}")

        return PredictionResponse(
            success=True,
            model="Best Trained Model",
            target="ttm_avg_rate",
            prediction=prediction,
            prediction_formatted=f"${prediction:.2f}/night",
            timestamp=datetime.now().isoformat(),
            request_id=request_id
        )
    except Exception as e:
        logger.error(f"Price prediction error: {e}")
        raise ValueError(f"Price prediction failed: {str(e)}")


@app.post("/api/predict/revenue", response_model=PredictionResponse)
async def predict_revenue(listing: ListingInput) -> PredictionResponse:
    try:
        request_id = f"revenue_{datetime.now().timestamp()}"
        model_name = "best_ttm_revenue_model"
        model = cache.load_model(model_name)
        features = cache.get_features("best_ttm_revenue")
        target_transform = cache.get_target_transform("best_ttm_revenue", model_name=model_name)

        X = prepare_single_prediction(listing, features)
        log_model_input_features("best_ttm_revenue", X, request_id)
        raw_prediction = float(model.predict(X)[0])
        prediction = float(np.maximum(np.expm1(raw_prediction), 0.0)) if target_transform == 'log1p' else raw_prediction

        logger.info(f"Revenue prediction: ${prediction:.2f}")
        
        return PredictionResponse(
            success=True,
            model="Best Revenue Model",
            target="ttm_revenue",
            prediction=prediction,
            prediction_formatted=f"${prediction:.2f}/month",
            timestamp=datetime.now().isoformat(),
            request_id=request_id
        )
    except Exception as e:
        logger.error(f"Revenue prediction error: {e}")
        raise ValueError(f"Revenue prediction failed: {str(e)}")


@app.post("/api/predict/ttm-avg-rate", response_model=PredictionResponse)
async def predict_ttm_avg_rate(listing: ListingInput) -> PredictionResponse:
    """Predict TTM average rate using best trained model from notebook."""
    try:
        request_id = f"ttm_avg_rate_{datetime.now().timestamp()}"
        model_name = "best_ttm_avg_rate_model"
        model = cache.load_model(model_name)
        features = cache.get_features("best_ttm_avg_rate")
        target_transform = cache.get_target_transform("best_ttm_avg_rate", model_name=model_name)

        X = prepare_single_prediction(listing, features)
        log_model_input_features("best_ttm_avg_rate", X, request_id)
        raw_prediction = float(model.predict(X)[0])
        prediction = float(np.maximum(np.expm1(raw_prediction), 0.0)) if target_transform == 'log1p' else raw_prediction

        logger.info(f"TTM average rate prediction: ${prediction:.2f}")
        
        return PredictionResponse(
            success=True,
            model="Best Trained Model",
            target="ttm_avg_rate",
            prediction=prediction,
            prediction_formatted=f"${prediction:.2f}",
            timestamp=datetime.now().isoformat(),
            request_id=request_id
        )
    except Exception as e:
        logger.error(f"TTM average rate prediction error: {e}")
        raise ValueError(f"TTM average rate prediction failed: {str(e)}")


@app.post("/api/predict/batch", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchListingInput) -> BatchPredictionResponse:
    from time import time
    start_time = time()
    
    try:
        model_map = {
            'price': ('model1_xgb_price', 'model1_price', 'ttm_avg_rate'),
            'revenue': ('best_ttm_revenue_model', 'best_ttm_revenue', 'ttm_revenue'),
            'ttm_avg_rate': ('best_ttm_avg_rate_model', 'best_ttm_avg_rate', 'ttm_avg_rate'),
        }
        
        if request.model not in model_map:
            raise ValueError(f"Unknown model: {request.model}")
        
        model_name, feature_key, target = model_map[request.model]
        
        model = cache.load_model(model_name)
        features = cache.get_features(feature_key)
        target_transform = cache.get_target_transform(feature_key, model_name=model_name)
        
        predictions = []
        errors = []
        successful = 0
        
        for idx, listing in enumerate(request.listings):
            try:
                X = prepare_single_prediction(listing, features)
                request_id = f"batch_{request.model}_{datetime.now().timestamp()}_{idx}"
                log_model_input_features(feature_key, X, request_id, listing_index=idx)
                raw_pred = float(model.predict(X)[0])
                pred = float(np.maximum(np.expm1(raw_pred), 0.0)) if target_transform == 'log1p' else raw_pred
                
                if request.model == 'price':
                    pred_formatted = f"${pred:.2f}/night"
                elif request.model == 'revenue':
                    pred_formatted = f"${pred:.2f}/month"
                else:
                    pred_formatted = f"${pred:.2f}"
                
                predictions.append({"listing_index": idx, "prediction": pred, "prediction_formatted": pred_formatted, "success": True})
                successful += 1
                
            except Exception as e:
                logger.warning(f"Batch prediction error for listing {idx}: {e}")
                errors.append({"listing_index": idx, "error": str(e)})
        
        processing_time = time() - start_time
        
        return BatchPredictionResponse(
            success=len(errors) == 0,
            model=request.model,
            total_listings=len(request.listings),
            successful_predictions=successful,
            failed_predictions=len(errors),
            predictions=predictions,
            errors=errors,
            processing_time_seconds=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise ValueError(f"Batch prediction failed: {str(e)}")


@app.post("/api/analyze/competitive", response_model=CompetitiveAnalysisResponse)
async def analyze_competitive_position(listing: ListingInput) -> CompetitiveAnalysisResponse:
    try:
        # Use the deployed price model for a consistent target-space estimate.
        model_name = "best_ttm_avg_rate_model"
        model = cache.load_model(model_name)
        features = cache.get_features("best_ttm_avg_rate")
        target_transform = cache.get_target_transform("best_ttm_avg_rate", model_name=model_name)

        X = prepare_single_prediction(listing, features)
        raw_prediction = float(model.predict(X)[0])
        predicted_price = float(np.maximum(np.expm1(raw_prediction), 0.0)) if target_transform == 'log1p' else raw_prediction

        ref_df, cluster_model = _load_competitive_reference()
        cluster_id = int(cluster_model.predict([[float(listing.latitude), float(listing.longitude)]])[0])
        cluster_df = ref_df[ref_df["cluster_id"] == cluster_id]
        if cluster_df.empty:
            cluster_df = ref_df

        cluster_size = int(len(cluster_df))
        cluster_median_price = float(cluster_df["ttm_avg_rate_business"].median())
        cluster_median_rating = float(cluster_df["avg_rating"].median())
        cluster_median_reviews = float(cluster_df["num_reviews"].median())
        cluster_median_amenities = float(cluster_df["amenities_count"].median())

        # Score components: cheaper than peers + quality signals.
        price_component = 100.0 * (cluster_median_price - predicted_price) / (cluster_median_price + 1e-6)
        rating_component = 20.0 * (float(listing.avg_rating) - cluster_median_rating)
        reviews_component = 10.0 * (float(listing.num_reviews) - cluster_median_reviews) / (cluster_median_reviews + 1e-6)
        amenities_component = 10.0 * (
            _compute_amenities_count(listing.amenities, listing.amenities_count) - cluster_median_amenities
        ) / (cluster_median_amenities + 1e-6)

        competitiveness_score = float(np.clip(price_component + rating_component + reviews_component + amenities_component, -100, 100))
        price_vs_cluster = float(predicted_price / (cluster_median_price + 1e-6))

        if competitiveness_score >= 30:
            market_position = "Strong"
        elif competitiveness_score >= 5:
            market_position = "Good"
        elif competitiveness_score >= -20:
            market_position = "Fair"
        else:
            market_position = "Weak"

        price_percentile = float((cluster_df["ttm_avg_rate_business"] <= predicted_price).mean())
        rating_percentile = float((cluster_df["avg_rating"] <= float(listing.avg_rating)).mean())
        reviews_percentile = float((cluster_df["num_reviews"] <= float(listing.num_reviews)).mean())

        return CompetitiveAnalysisResponse(
            success=True,
            cluster_id=cluster_id,
            cluster_size=cluster_size,
            market_position=market_position,
            competitiveness_score=competitiveness_score,
            price_vs_cluster=price_vs_cluster,
            predicted_price=predicted_price,
            cluster_median_price=cluster_median_price,
            percentile_ranks={
                "price_percentile": price_percentile,
                "rating_percentile": rating_percentile,
                "reviews_percentile": reviews_percentile,
            },
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        logger.error(f"Competitive analysis error: {e}")
        raise ValueError(f"Competitive analysis failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
