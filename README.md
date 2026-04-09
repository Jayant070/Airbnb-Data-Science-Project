# Airbnb Pricing and Revenue Intelligence

End-to-end project for Airbnb market intelligence:

- data processing and feature engineering
- model training and artifact registry
- FastAPI serving for inference
- Streamlit dashboard for interactive use

## Current Project Status

The active production artifacts are:

- `models/best_ttm_avg_rate_model.pkl`
- `models/best_ttm_revenue_model.pkl`
- `models/best_ttm_avg_rate_features.txt`
- `models/best_ttm_revenue_features.txt`
- `models/model_registry.json`

Both active targets use log-transformed training targets (`log1p`) and are inverse-transformed to business scale during API inference.

## Repository Layout

- `api/`: FastAPI app and API container files
- `app.py`: Streamlit dashboard
- `data/`: raw and processed datasets
- `scripts/`: data prep, geo encoding, training utilities
- `src/`: reusable preprocessing and training pipelines
- `models/`: trained artifacts and model registry
- `docs/`: metrics summary and local inference helper
- `Notebooks/`: EDA, feature engineering, training, and clustering notebooks

## Local Setup

Create an environment and install project dependencies:

```bash
pip install -r requirements.txt
```

Install API dependencies:

```bash
pip install -r api/requirements.txt
```

## Run API

```bash
cd api
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Docs UI:

```text
http://localhost:8000/api/docs
```

## Run Dashboard

From project root:

```bash
streamlit run app.py
```

## Main API Endpoints

- `GET /api/health`
- `POST /api/predict/price`
- `POST /api/predict/revenue`
- `POST /api/predict/ttm-avg-rate`
- `POST /api/predict/batch`
- `POST /api/analyze/competitive`

## Notes

- The API performs feature engineering internally, including amenity parsing and geo-based enrichment.
- Request-level `distance_from_city_center` and `city_population` override geo-resolved values when provided.
- Use `DEBUG_FEATURE_LOGGING=true|false` to control final model input logging.
