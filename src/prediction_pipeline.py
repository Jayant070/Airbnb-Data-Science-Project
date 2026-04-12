"""
Prediction Pipeline Module

Handles feature engineering, transformations, and prediction preparation
for the Airbnb pricing models.
"""

import logging
import sys
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "scripts" / "geoencoding"))
sys.path.insert(0, str(project_root / "scripts" / "data" / "main"))

from geo_encoding import resolve_city_distance
from amenties_utility import create_amenity_features
from listing_type_utils import categorize_listing

logger = logging.getLogger(__name__)
TRAINING_NUM_REVIEWS_MAX = 399.0
TRAINING_DISTANCE_FROM_CITY_CENTER_MAX = 16.0
DEBUG_FEATURE_LOGGING = False

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

def normalize_key(value: str) -> str:
    """Normalize string key to lowercase with underscores."""
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")

def normalize_listing_type(value: str) -> str:
    """
    Normalize listing type to match training categories.
    
    Maps raw input to the categories used in model training:
    - apartment, home, condo, villa, bungalow, loft
    - hospitality, nature_stay, unique_stay, luxury_unique
    - other (for unmapped types)
    """
    category = categorize_listing(value)
    return category if category else "other"

def normalize_amenities_input(amenities: Optional[Union[str, List[str]]]) -> str:
    """Standardize amenities input to comma-separated string."""
    if amenities is None:
        return ""
    if isinstance(amenities, list):
        return ", ".join(str(a).strip() for a in amenities if str(a).strip())

    text = str(amenities)
    for sep in [";", "|", "\n", "\t", "/"]:
        text = text.replace(sep, ",")

    parts = [p.strip() for p in text.split(",") if p.strip()]
    return ", ".join(parts)

def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float with fallback default."""
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default

# Encode cancellation policy to numeric value 
def encode_cancellation_policy(value: Union[str, float, int]) -> float:
    """Encode cancellation policy to numeric value."""
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(max(0.0, value))

    text = str(value or "").strip()
    if not text:
        return 4.0

    if text in CANCELLATION_ORDER:
        return float(CANCELLATION_ORDER[text])

    lowered = text.lower()
    for key, encoded in CANCELLATION_ORDER.items():
        if key.lower() == lowered:
            return float(encoded)

    try:
        return float(max(0.0, float(text)))
    except (TypeError, ValueError):
        logger.warning(f"Unknown cancellation policy '{text}', defaulting to Flexible (4)")
        return 4.0

# Compute amenities count from either list or string input
def compute_amenities_count(amenities: Optional[Union[str, List[str]]], fallback_count: Optional[int]) -> int:
    """Compute count of amenities from input."""
    if isinstance(amenities, list):
        cleaned = [str(a).strip() for a in amenities if str(a).strip()]
        if cleaned:
            return len(cleaned)

    if isinstance(amenities, str):
        normalized = normalize_amenities_input(amenities)
        if normalized:
            return len([item for item in normalized.split(",") if item.strip()])

    if fallback_count is not None:
        return int(max(0, fallback_count))

    return 0

# Apply log1p transformation to match training feature scale
def apply_training_feature_transforms(df: pd.DataFrame, expected_features: List[str]) -> pd.DataFrame:
    max_user_reviews = 399.0
    max_user_distance = 16.0
    
    if "num_reviews" in expected_features and "num_reviews" in df.columns:
        df["num_reviews"] = np.log1p(np.clip(df["num_reviews"], a_min=0.0, a_max=max_user_reviews))
    
    if "distance_from_city_center" in expected_features and "distance_from_city_center" in df.columns:
        df["distance_from_city_center"] = np.log1p(np.clip(df["distance_from_city_center"], a_min=0.0, a_max=max_user_distance))
    
    return df

def engineer_features(listing_dict: Dict[str, Any], expected_features: List[str]) -> pd.DataFrame:
    """
    Feature engineering pipeline:
    1. Parse raw input
    2. Add categorical encodings (one-hot encoding)
    3. Calculate derived features
    4. Determine geographic zone
    Returns DataFrame with all engineered features
    """
    latitude = float(listing_dict['latitude'])
    longitude = float(listing_dict['longitude'])
    cancellation_policy_encoded = encode_cancellation_policy(listing_dict.get('cancellation_policy'))
    amenities_count = compute_amenities_count(listing_dict.get('amenities'), listing_dict.get('amenities_count'))
    provided_distance = listing_dict.get('distance_from_city_center')
    provided_city_population = listing_dict.get('city_population')

    zone = 'Northern America'
    resolved_distance = 0.0
    resolved_city_population = 0.0

    try:
        geo_result = resolve_city_distance(
            cityname=None,
            latitude=latitude,
            longitude=longitude,
        )
        zone = geo_result.get('zone') or 'Northern America'
        resolved_distance = safe_float(geo_result.get('distance_from_city_center', geo_result.get('distance_km')), 0.0)
        resolved_city_population = safe_float(geo_result.get('city_population'), 0.0)
    except Exception as e:
        logger.warning(f"Failed to resolve city distance/zone via geoencoding: {e}")

    distance_from_city_center = safe_float(provided_distance, resolved_distance)
    city_population = safe_float(provided_city_population, resolved_city_population)

    amenities_text = normalize_amenities_input(listing_dict.get('amenities'))
    amenity_features: Dict[str, Any] = {}
    if amenities_text:
        try:
            amenity_df = create_amenity_features(pd.Series([amenities_text]))
            amenity_features = amenity_df.iloc[0].to_dict()
        except Exception as e:
            logger.warning(f"Failed to parse amenities using amenities utility: {e}")

    total_amenities = safe_float(
        amenity_features.get('total_amenities')
        if 'total_amenities' in amenity_features
        else (
            listing_dict.get('total_amenities')
            if listing_dict.get('total_amenities') is not None
            else amenities_count
        )
    )

    basic_count = safe_float(
        amenity_features.get('basic_count')
        if 'basic_count' in amenity_features
        else (listing_dict.get('basic_count') if listing_dict.get('basic_count') is not None else total_amenities * 0.20)
    )
    comfort_count = safe_float(
        amenity_features.get('comfort_count')
        if 'comfort_count' in amenity_features
        else (listing_dict.get('comfort_count') if listing_dict.get('comfort_count') is not None else total_amenities * 0.20)
    )
    kitchen_count = safe_float(
        amenity_features.get('kitchen_count')
        if 'kitchen_count' in amenity_features
        else (listing_dict.get('kitchen_count') if listing_dict.get('kitchen_count') is not None else total_amenities * 0.15)
    )
    safety_count = safe_float(
        amenity_features.get('safety_count')
        if 'safety_count' in amenity_features
        else (listing_dict.get('safety_count') if listing_dict.get('safety_count') is not None else total_amenities * 0.15)
    )
    outdoor_count = safe_float(
        amenity_features.get('outdoor_count')
        if 'outdoor_count' in amenity_features
        else (listing_dict.get('outdoor_count') if listing_dict.get('outdoor_count') is not None else total_amenities * 0.10)
    )
    family_count = safe_float(
        amenity_features.get('family_count')
        if 'family_count' in amenity_features
        else (listing_dict.get('family_count') if listing_dict.get('family_count') is not None else total_amenities * 0.10)
    )
    services_count = safe_float(
        amenity_features.get('services_count')
        if 'services_count' in amenity_features
        else (listing_dict.get('services_count') if listing_dict.get('services_count') is not None else total_amenities * 0.10)
    )

    comfort_to_total_ratio = listing_dict.get('comfort_to_total_ratio')
    if comfort_to_total_ratio is None:
        comfort_to_total_ratio = safe_float(
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
    }

    df = pd.DataFrame([feature_row])

    expected_listing_type_cols = [f for f in expected_features if f.startswith('listing_type_')]
    expected_room_type_cols = [f for f in expected_features if f.startswith('room_type_')]
    expected_zone_cols = [f for f in expected_features if f.startswith('geographic_zone_')]

    listing_type_value = normalize_listing_type(listing_dict.get('listing_type', 'other'))
    room_type_value = normalize_key(listing_dict.get('room_type', 'private_room'))
    zone_value = str(zone).strip()

    for col in expected_listing_type_cols:
        col_value = normalize_key(col.replace('listing_type_', ''))
        df[col] = 1 if col_value == listing_type_value else 0

    if expected_listing_type_cols and int(df[expected_listing_type_cols].to_numpy().sum()) == 0 and 'listing_type_other' in df.columns:
        df['listing_type_other'] = 1

    for col in expected_room_type_cols:
        col_value = normalize_key(col.replace('room_type_', ''))
        df[col] = 1 if col_value == room_type_value else 0

    for col in expected_zone_cols:
        zone_label = col.replace('geographic_zone_', '').strip()
        df[col] = 1 if zone_label == zone_value else 0

    if expected_zone_cols and int(df[expected_zone_cols].to_numpy().sum()) == 0 and 'geographic_zone_Northern America' in df.columns:
        df['geographic_zone_Northern America'] = 1

    for col in expected_features:
        if col not in df.columns:
            df[col] = 0.0

    df = pd.DataFrame(df.loc[:, expected_features])
    return apply_training_feature_transforms(df, expected_features)

# Log final engineered feature values sent to the model
def log_model_input_features(model_key: str, X: pd.DataFrame, request_id: str, listing_index: Optional[int] = None) -> None:
    
    if not DEBUG_FEATURE_LOGGING:
        return

    if X.empty:
        logger.warning(f"[{request_id}] Empty feature frame for {model_key}")
        return

    suffix = f" listing_index={listing_index}" if listing_index is not None else ""
    feature_map = {col: X.iloc[0][col] for col in X.columns}
    logger.info(f"[{request_id}] Final model input features for {model_key}{suffix}: {feature_map}")

# Prepare features for prediction with full feature engineering and debugging
def prepare_prediction(listing_dict: Dict[str, Any], features: List[str], request_id: str = "") -> pd.DataFrame:
    df = engineer_features(listing_dict, features)
    
    nan_cols = df.columns[df.isna().any()].tolist()
    if nan_cols:
        raise ValueError(f"NaN values found in features: {nan_cols}")
    
    if DEBUG_FEATURE_LOGGING:
        logger.info(f"[{request_id}] Feature shape: {df.shape}")
        logger.info(f"[{request_id}] num_reviews in features: {'num_reviews' in df.columns}")
        if 'num_reviews' in df.columns:
            logger.info(f"[{request_id}] num_reviews value: {df['num_reviews'].iloc[0]:.6f}")
        if 'distance_from_city_center' in df.columns:
            logger.info(f"[{request_id}] distance_from_city_center value: {df['distance_from_city_center'].iloc[0]:.6f}")
        logger.info(f"[{request_id}] Feature columns: {list(df.columns)[:15]}...")
    
    return df

def make_prediction(listing_dict: Dict[str, Any], model_name: str, feature_key: str, 
                    cache: Any, request_id: str = "") -> float:
    """
    End-to-end prediction pipeline: load model, prepare features, predict, and return business-scale value.
    
    WORKFLOW:
    1. Load the trained model from cache
    2. Get the list of features the model expects
    3. Get the target transformation metadata (log1p or none)
    4. Prepare features by engineering and transforming them to match training scale
    5. Make the raw prediction (on log scale if target is log1p-transformed)
    6. Apply inverse transformation to convert back to business scale (dollars)
    7. Return the final prediction
    
    Args:
        listing_dict: Raw user listing input (bedrooms, amenities, etc.)
        model_name: Name of the model file (e.g., 'best_ttm_revenue_model')
        feature_key: Key in model registry (e.g., 'best_ttm_revenue')
        cache: ModelCache instance from main.py that loads models and metadata
        request_id: Optional request ID for logging
    
    Returns:
        float: Final prediction in business scale (dollars per year/month/night)
    
    Example:
        >>> prediction = make_prediction(
        ...     listing_dict={'bedrooms': 2, 'num_reviews': 67, ...},
        ...     model_name='best_ttm_revenue_model',
        ...     feature_key='best_ttm_revenue',
        ...     cache=cache_obj,
        ...     request_id='revenue_1234567890'
        ... )
    """
    
    logger.info(f"[{request_id}] Loading model: {model_name}")
    model = cache.load_model(model_name)
    
    logger.info(f"[{request_id}] Retrieving {len(cache.get_features(feature_key))} required features for {feature_key}")
    features = cache.get_features(feature_key)
    
    target_transform = cache.get_target_transform(feature_key, model_name=model_name)
    logger.info(f"[{request_id}] Target transform: {target_transform}")
    
    logger.info(f"[{request_id}] Preparing features via engineering pipeline")
    X = prepare_prediction(listing_dict, features, request_id)
    
    log_model_input_features(feature_key, X, request_id)
    
    logger.info(f"[{request_id}] Making prediction with {len(features)} features")
    
    try:
        from xgboost import Booster
        if isinstance(model, Booster):
            from xgboost import DMatrix
            logger.info(f"[{request_id}] Creating DMatrix from {X.shape} features with column names")
            dmatrix = DMatrix(X.values, feature_names=list(X.columns))
            logger.info(f"[{request_id}] DMatrix created successfully, predicting...")
            raw_prediction = float(model.predict(dmatrix)[0])
            logger.info(f"[{request_id}] Used XGBoost Booster (JSON format) for prediction")
        else:
            raw_prediction = float(model.predict(X)[0])
    except ImportError:
        raw_prediction = float(model.predict(X)[0])
    except Exception as e:
        logger.error(f"[{request_id}] Prediction error: {type(e).__name__}: {str(e)}")
        raise
    
    logger.info(f"[{request_id}] Raw model prediction (log scale): {raw_prediction}")
    
    if target_transform == 'log1p':
        prediction = float(np.maximum(np.expm1(raw_prediction), 0.0))
        logger.info(f"[{request_id}] Applied expm1 inverse transform: {raw_prediction} → ${prediction:.2f}")
    else:
        prediction = raw_prediction
        logger.info(f"[{request_id}] No transform applied; using raw prediction: ${prediction:.2f}")
    
    logger.info(f"[{request_id}] Final prediction: ${prediction:.2f}")
    return prediction
