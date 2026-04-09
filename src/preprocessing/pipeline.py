from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from scripts.data.main.amenties_utility import create_amenity_features
from scripts.data.main.listing_type_utils import categorize_listing


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


def _convert_bool_like(df: pd.DataFrame) -> pd.DataFrame:
    bool_like_map = {
        "TRUE": 1,
        "FALSE": 0,
        "true": 1,
        "false": 0,
        "True": 1,
        "False": 0,
        True: 1,
        False: 0,
        "YES": 1,
        "NO": 0,
        "yes": 1,
        "no": 0,
        "Y": 1,
        "N": 0,
    }

    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)
            continue

        if df[col].dtype == object:
            s = df[col].dropna().astype(str).str.strip()
            if len(s) == 0:
                continue
            unique_vals = set(s.unique())
            allowed = {"TRUE", "FALSE", "true", "false", "True", "False", "YES", "NO", "yes", "no", "Y", "N"}
            if unique_vals.issubset(allowed):
                df[col] = df[col].map(bool_like_map)

    return df


def run_feature_engineering_pipeline(
    input_path: Path,
    output_path: Path,
    *,
    log_transform_target: bool = True,
) -> pd.DataFrame:
    df = pd.read_csv(input_path, low_memory=False)

    # Step 1: missing values
    missing_ratio = df.isna().mean()
    too_missing_cols = missing_ratio[missing_ratio > 0.50].index.tolist()
    df = df.drop(columns=too_missing_cols, errors="ignore")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if df[col].isna().any():
            mode_series = df[col].mode(dropna=True)
            if not mode_series.empty:
                df[col] = df[col].fillna(mode_series.iloc[0])

    # Step 2: drop leakage/useless
    drop_always = [
        "listing_id", "cover_photo_url", "host_id", "host_name", "listing_name", "cohost_names", "currency",
        "city", "country", "country_code",
    ]
    drop_l90d = [col for col in df.columns if col.startswith("l90d_")]
    drop_native = [col for col in df.columns if col.endswith("_native")]
    drop_leakage = ["ttm_revenue", "ttm_revpar", "ttm_occupancy", "ttm_adjusted_occupancy", "ttm_adjusted_revpar"]
    cols_to_drop = [col for col in (drop_always + drop_l90d + drop_native + drop_leakage) if col in df.columns]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    # Step 3: derived features
    if "amenities" in df.columns:
        amenity_features = create_amenity_features(df["amenities"])
        df = pd.concat([df, amenity_features], axis=1)
        df = df.drop(columns=["amenities"], errors="ignore")

    rating_cols = [c for c in df.columns if c.startswith("rating_")]
    if rating_cols:
        df["avg_rating"] = df[rating_cols].mean(axis=1, skipna=True)
        df = df.drop(columns=rating_cols, errors="ignore")

    if "listing_type" in df.columns:
        df["listing_type"] = df["listing_type"].apply(categorize_listing)
        if "luxury_score" in df.columns:
            df.loc[df["listing_type"] == "luxury_unique", "luxury_score"] += 8

    # Step 4: bool-like encoding
    df = _convert_bool_like(df)

    # Step 5: invalid rows and robust outlier filtering
    for col in ["min_nights", "distance_from_city_center", "cleaning_fee", "bedrooms", "beds", "baths", "guests", "num_reviews", "photos_count"]:
        if col in df.columns:
            df = df[df[col] >= 0]

    outlier_features = {
        "distance_from_city_center": 3.0,
        "min_nights": 3.0,
        "cleaning_fee": 3.0,
        "bedrooms": 3.0,
        "beds": 3.0,
        "baths": 3.0,
        "num_reviews": 3.0,
        "photos_count": 3.0,
        "ttm_blocked_days": 3.0,
    }
    keep_mask = pd.Series(True, index=df.index)
    for col, iqr_mult in outlier_features.items():
        if col not in df.columns:
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - iqr_mult * iqr
        upper = q3 + iqr_mult * iqr
        keep_mask &= ~((df[col] < lower) | (df[col] > upper))
    df = df[keep_mask].copy()

    # Step 6: categorical encoding
    if "cancellation_policy" in df.columns:
        df["cancellation_policy"] = df["cancellation_policy"].astype(str).str.strip().map(CANCELLATION_ORDER).fillna(4)

    encode_cols = [col for col in ["listing_type", "room_type", "geographic_zone"] if col in df.columns]
    if encode_cols:
        df = pd.get_dummies(df, columns=encode_cols)

    # Step 7: log transforms
    log_cols = ["num_reviews", "distance_from_city_center"]
    if log_transform_target:
        log_cols.extend(["ttm_avg_rate", "ttm_revenue"])
    for col in log_cols:
        if col in df.columns:
            df[col] = np.log1p(np.clip(df[col], a_min=0.0, a_max=None))

    # Step 8: duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df
