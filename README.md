# Airbnb Pricing and Revenue Intelligence

An end-to-end Data Science and Machine Learning platform designed to estimate Airbnb property pricing and revenue potential. The project combines data preprocessing, feature engineering, predictive modeling, API deployment, and interactive analytics to provide actionable market intelligence for property owners and investors.

## Project Overview

This project analyzes Airbnb listing data and predicts:

* Average nightly rental price
* Estimated annual revenue
* Market competitiveness
* Pricing opportunities across locations

The system includes machine learning models, a FastAPI inference service, and a Streamlit dashboard for real-time business insights.

## Features

* Data cleaning and preprocessing pipeline
* Feature engineering and geo-based enrichment
* Machine Learning model training and evaluation
* Model registry and artifact management
* FastAPI-based prediction service
* Interactive Streamlit dashboard
* Batch prediction support
* Competitive market analysis

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-Learn
* FastAPI
* Streamlit
* Joblib
* Machine Learning
* Data Visualization

## Project Architecture

```text
Raw Airbnb Data
        │
        ▼
Data Cleaning & Feature Engineering
        │
        ▼
Model Training & Evaluation
        │
        ▼
Model Registry
        │
 ┌──────┴──────┐
 ▼             ▼
FastAPI      Streamlit
Inference    Dashboard
```

## Repository Structure

```text
api/        → FastAPI prediction service
src/        → Training and preprocessing pipelines
scripts/    → Data preparation utilities
models/     → Trained model artifacts
data/       → Processed datasets
docs/       → Documentation and metrics
notebooks/  → EDA and experimentation
app.py      → Streamlit dashboard
```

## Running the Project

### Install Dependencies

```bash
pip install -r requirements.txt
pip install -r api/requirements.txt
```

### Start FastAPI Server

```bash
cd api
uvicorn main:app --reload
```

### Launch Dashboard

```bash
streamlit run app.py
```

## API Endpoints

* GET `/api/health`
* POST `/api/predict/price`
* POST `/api/predict/revenue`
* POST `/api/predict/ttm-avg-rate`
* POST `/api/predict/batch`
* POST `/api/analyze/competitive`

## Future Improvements

* Deep learning-based pricing models
* Automated model retraining pipeline
* Cloud deployment with CI/CD
* Time-series demand forecasting
* Recommendation engine for hosts

## Contributors

* Jayant Jadhav
* Rushi K.
