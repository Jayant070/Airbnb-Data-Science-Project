# Project Documentation

This folder contains supporting documentation and helper utilities aligned with the current model stack.

## Files in This Folder

- `ALL_MODELS_METRICS.txt`: Current metrics summary from `models/model_registry.json`
- `model_inference.py`: Standalone local inference helper for loading current model artifacts

## Source of Truth

Model metadata is maintained in:

- `models/model_registry.json`

Model artifacts and features are in:

- `models/best_ttm_avg_rate_model.pkl`
- `models/best_ttm_avg_rate_features.txt`
- `models/best_ttm_revenue_model.pkl`
- `models/best_ttm_revenue_features.txt`

## Notes

- API inference in `api/main.py` is the production path.
- `docs/model_inference.py` is a local helper for notebook or script experimentation.
- Targets are trained with `log1p` and can be inverse-transformed to business scale for reporting.
