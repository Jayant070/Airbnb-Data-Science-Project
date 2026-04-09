# Airbnb Prediction API

FastAPI service for Airbnb pricing and revenue inference with shared feature engineering.

## What This API Serves

- `ttm_avg_rate` predictions from `models/best_ttm_avg_rate_model.pkl`
- `ttm_revenue` predictions from `models/best_ttm_revenue_model.pkl`
- Batch inference for multiple listings
- Competitive position analysis based on geo clusters

All prediction outputs are returned on business scale. If a model target was trained with `log1p`, the API applies inverse transform (`expm1`) before returning values.

## Run Locally

From the `api` folder:

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open Swagger UI:

```text
http://localhost:8000/api/docs
```

## Endpoints

- `GET /api/health`
- `POST /api/predict/price`
- `POST /api/predict/revenue`
- `POST /api/predict/ttm-avg-rate`
- `POST /api/predict/batch`
- `POST /api/analyze/competitive`

## Minimal Prediction Request

```json
{
  "bedrooms": 2,
  "beds": 3,
  "baths": 1,
  "guests": 4,
  "photos_count": 50,
  "superhost": 1,
  "num_reviews": 100,
  "avg_rating": 4.8,
  "latitude": 40.7128,
  "longitude": -74.006,
  "cancellation_policy": "Flexible",
  "min_nights": 2,
  "cleaning_fee": 90,
  "extra_guest_fee": 20,
  "registration": 1,
  "professional_management": 0
}
```

## Feature Engineering Behavior

- If `amenities` is provided, amenity features are generated using `scripts/data/main/amenties_utility.py`.
- `listing_type` is normalized using `scripts/data/main/listing_type_utils.py`.
- Geo fields are resolved from coordinates using `scripts/geoencoding/geo_encoding.py`.
- If `distance_from_city_center` or `city_population` is provided in request, those values override resolved values.

## Model File Requirements

The service expects these files in `../models`:

- `best_ttm_avg_rate_model.pkl`
- `best_ttm_avg_rate_features.txt`
- `best_ttm_revenue_model.pkl`
- `best_ttm_revenue_features.txt`
- `model_registry.json`

Legacy compatibility remains in code for old names (`model1_xgb_price`, `model2_xgb_revenue`) where possible.

## Optional Environment Variables

- `DEBUG_FEATURE_LOGGING=true|false` (default: `true`)
  - Logs final engineered model input features for each prediction.

## Docker

From the `api` folder:

```bash
docker compose up --build
```
