"""
Streamlit Dashboard for Airbnb model serving.
Supports Price Prediction, Revenue Prediction, and Competitive Analysis.
"""

from datetime import datetime
from pathlib import Path
import sys
from typing import Dict, Tuple

import plotly.graph_objects as go
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "scripts" / "geoencoding"))
from geo_encoding import resolve_city_distance

st.set_page_config(
    page_title="Airbnb Intelligence Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --brand: #0f766e;
        --ink: #0f172a;
        --muted: #64748b;
        --ok: #166534;
        --ok-bg: #dcfce7;
        --err: #991b1b;
        --err-bg: #fee2e2;
    }
    .app-shell {
        background: radial-gradient(1200px 500px at 10% -20%, #99f6e4 0%, rgba(153,246,228,0) 60%),
                    radial-gradient(1000px 500px at 100% 0%, #bfdbfe 0%, rgba(191,219,254,0) 55%);
        border-radius: 20px;
        padding: 22px;
        border: 1px solid #e2e8f0;
        margin-bottom: 12px;
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        color: var(--ink);
        margin-bottom: 6px;
        letter-spacing: -0.02em;
    }
    .hero-sub {
        color: var(--muted);
        font-size: 1rem;
        margin-bottom: 0;
    }
    .result-ok {
        background: var(--ok-bg);
        border-left: 5px solid var(--ok);
        border-radius: 10px;
        padding: 14px;
        color: #14532d;
    }
    .result-err {
        background: var(--err-bg);
        border-left: 5px solid var(--err);
        border-radius: 10px;
        padding: 14px;
        color: #7f1d1d;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

API_BASE_URL = st.secrets.get("api_url", "http://localhost:8000")
API_ENDPOINTS = {
    "health": f"{API_BASE_URL}/api/health",
    "price": f"{API_BASE_URL}/api/predict/price",
    "revenue": f"{API_BASE_URL}/api/predict/revenue",
    "competitive": f"{API_BASE_URL}/api/analyze/competitive",
}

CANCELLATION_LABELS = [
    "Full Refundable Until Check-in",
    "Full Refundable Until 24 Hours Before Check-in",
    "Full Refundable Until 72 Hours Before Check-in",
    "Refundable",
    "Flexible",
    "Moderate",
    "Limited",
    "Firm",
    "Strict",
    "Non-refundable",
    "Super Strict 30 Days",
    "Super Strict 60 Days",
]

PREDEFINED_AMENITIES = [
    "Wifi", "Kitchen", "Air conditioning", "Heating", "Refrigerator", "Essentials",
    "Portable fans", "Microwave", "Stove", "Oven", "Coffee maker", "Cooking basics",
    "Dishes and silverware", "Smoke alarm", "Fire extinguisher", "First aid kit",
    "Carbon monoxide alarm", "Patio or balcony", "Backyard", "Outdoor furniture", "Hammock",
    "Waterfront", "Lake access", "Crib", "High chair", "Children's dinnerware", "Baby bath",
    "Board games", "Long term stays allowed", "Luggage dropoff allowed", "Cleaning before checkout",
    "Outdoor kitchen", "Sauna", "Resort access", "Washer", "Dryer", "Free parking on premises",
    "Dedicated workspace", "Pool", "Hot tub", "Gym", "Beach access", "Pets allowed", "TV",
    "Elevator", "Balcony", "BBQ grill",
]


def check_api_health() -> Tuple[bool, Dict]:
    try:
        response = requests.get(API_ENDPOINTS["health"], timeout=5)
        if response.status_code == 200:
            return True, response.json()
        return False, {"error": response.text}
    except Exception as exc:
        return False, {"error": str(exc)}


def call_api(endpoint: str, data: Dict) -> Dict:
    try:
        response = requests.post(endpoint, json=data, timeout=30)
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        return {"success": False, "error": response.text}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Connection failed. Is FastAPI running on localhost:8000?"}
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timed out. Please try again."}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def resolve_geo_preview(latitude: float, longitude: float) -> Dict:
    try:
        geo = resolve_city_distance(cityname=None, latitude=latitude, longitude=longitude)
        return {
            "nearest_city": str(geo.get("name") or "Unknown"),
            "zone": str(geo.get("zone") or "Unknown"),
            "distance_from_city_center": float(geo.get("distance_from_city_center") or geo.get("distance_km") or 0.0),
            "city_population": float(geo.get("city_population") or 0.0),
            "error": "",
        }
    except Exception as exc:
        return {
            "nearest_city": "Unknown",
            "zone": "Unknown",
            "distance_from_city_center": 0.0,
            "city_population": 0.0,
            "error": str(exc),
        }


def build_price_gauge(value: float, title: str) -> go.Figure:
    low = max(0.0, value * 0.75)
    high = value * 1.25
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"prefix": "$", "valueformat": ".2f"},
            title={"text": title},
            gauge={
                "axis": {"range": [low, high]},
                "bar": {"color": "#0f766e"},
                "steps": [
                    {"range": [low, value * 0.9], "color": "#dbeafe"},
                    {"range": [value * 0.9, value * 1.1], "color": "#bfdbfe"},
                    {"range": [value * 1.1, high], "color": "#93c5fd"},
                ],
            },
        )
    )
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def build_listing_inputs(prefix: str) -> Dict:
    c1, c2 = st.columns(2)
    with c1:
        listing_type = st.text_input("Listing Type", value="House", key=f"{prefix}_listing_type")
        bedrooms = st.number_input("Bedrooms", min_value=0, max_value=50, value=2, step=1, key=f"{prefix}_bedrooms")
        baths = st.number_input("Bathrooms", min_value=0, max_value=50, value=1, step=1, key=f"{prefix}_baths")
        photos_count = st.number_input("Photos Count", min_value=0, max_value=500, value=50, key=f"{prefix}_photos")
        avg_rating = st.number_input("Average Rating", min_value=1.0, max_value=5.0, value=4.8, step=0.1, key=f"{prefix}_rating")
        min_nights = st.number_input("Minimum Nights", min_value=0, max_value=1000, value=2, key=f"{prefix}_min_nights")
    with c2:
        room_type = st.selectbox("Room Type", ["entire_home", "hotel_room", "private_room", "shared_room"], index=2, key=f"{prefix}_room_type")
        beds = st.number_input("Beds", min_value=0, max_value=50, value=3, step=1, key=f"{prefix}_beds")
        guests = st.number_input("Guests", min_value=0, max_value=100, value=4, step=1, key=f"{prefix}_guests")
        num_reviews = st.number_input("Number of Reviews", min_value=0, max_value=100000, value=100, key=f"{prefix}_reviews")
        superhost = st.selectbox("Superhost", ["No", "Yes"], index=1, key=f"{prefix}_superhost")
        cancellation_policy = st.selectbox("Cancellation Policy", CANCELLATION_LABELS, index=CANCELLATION_LABELS.index("Flexible"), key=f"{prefix}_cancel")

    st.markdown("#### Location and Fees")
    c3, c4 = st.columns(2)
    with c3:
        latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=40.7128, format="%.6f", key=f"{prefix}_lat")
        cleaning_fee = st.number_input("Cleaning Fee ($)", min_value=0.0, max_value=10000.0, value=100.0, step=1.0, key=f"{prefix}_cleaning")
    with c4:
        longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=-74.0060, format="%.6f", key=f"{prefix}_lon")
        extra_guest_fee = st.number_input("Extra Guest Fee ($)", min_value=0.0, max_value=5000.0, value=20.0, step=1.0, key=f"{prefix}_extra_guest")

    geo = resolve_geo_preview(float(latitude), float(longitude))
    g1, g2, g3 = st.columns(3)
    with g1:
        st.metric("Nearest City", geo["nearest_city"])
    with g2:
        st.metric("Distance from Center", f"{geo['distance_from_city_center']:.2f} km")
    with g3:
        st.metric("City Population", f"{geo['city_population']:,.0f}")

    c5, c6 = st.columns(2)
    with c5:
        registration = st.selectbox("Registration", ["Not Registered", "Registered"], index=1, key=f"{prefix}_registration")
    with c6:
        professional_management = st.selectbox("Professional Management", ["No", "Yes"], index=0, key=f"{prefix}_pro_mgmt")

    with st.expander("Advanced Optional Inputs", expanded=False):
        city_name = st.text_input("City Name (optional)", value="New York", key=f"{prefix}_city_name")
        c7, c8 = st.columns(2)
        with c7:
            ttm_blocked_days = st.number_input("TTM Blocked Days", min_value=0.0, value=0.0, step=1.0, key=f"{prefix}_ttm_blocked")
        with c8:
            ttm_total_days = st.number_input("TTM Total Days", min_value=1.0, value=365.0, step=1.0, key=f"{prefix}_ttm_total")

    st.markdown("#### Amenities")
    amenities = st.multiselect(
        "Select amenities",
        options=PREDEFINED_AMENITIES,
        default=["Wifi", "Kitchen", "Air conditioning", "Dedicated workspace"],
        placeholder="Start typing to filter amenity suggestions",
        key=f"{prefix}_amenities",
    )
    custom_amenities = st.text_input(
        "Custom amenities (optional)",
        value="",
        placeholder="Example: Cable, Smart Lock",
        key=f"{prefix}_custom_amenities",
    )

    custom_list = [item.strip() for item in custom_amenities.split(",") if item.strip()]
    all_amenities = amenities + [item for item in custom_list if item not in amenities]

    payload = {
        "bedrooms": int(bedrooms),
        "beds": int(beds),
        "baths": int(baths),
        "guests": int(guests),
        "photos_count": int(photos_count),
        "superhost": 1 if superhost == "Yes" else 0,
        "num_reviews": int(num_reviews),
        "avg_rating": float(avg_rating),
        "latitude": float(latitude),
        "longitude": float(longitude),
        "distance_from_city_center": float(geo["distance_from_city_center"]),
        "city_population": float(geo["city_population"]),
        "cancellation_policy": cancellation_policy,
        "min_nights": int(min_nights),
        "cleaning_fee": float(cleaning_fee),
        "extra_guest_fee": float(extra_guest_fee),
        "registration": 1 if registration == "Registered" else 0,
        "professional_management": 1 if professional_management == "Yes" else 0,
        "listing_type": listing_type,
        "room_type": room_type,
        "city_name": city_name,
        "amenities": all_amenities,
        "ttm_blocked_days": float(ttm_blocked_days),
        "ttm_total_days": float(ttm_total_days),
    }
    return payload


with st.sidebar:
    st.title("🏠 Airbnb Intelligence")
    st.caption("Price, Revenue, and Competitive analytics")
    if st.button("↻", help="Refresh API status"):
        st.rerun()

    is_healthy, health_data = check_api_health()
    if is_healthy:
        st.success("API Connected")
    else:
        st.error("API Unavailable")
    st.caption(f"Base URL: {API_BASE_URL}")

    page = st.radio(
        "Select Page",
        ["Price Prediction", "Revenue Prediction", "Competitive Analysis"],
    )

st.markdown(
    """
    <div class="app-shell">
      <div class="hero-title">Airbnb Intelligence Dashboard</div>
      <p class="hero-sub">Predict rate/revenue and analyze local competitive position from one interface.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not is_healthy:
    st.markdown('<div class="result-err">FastAPI is not reachable. Start API and refresh.</div>', unsafe_allow_html=True)
    st.stop()

left_col, right_col = st.columns([1.2, 1.0], gap="large")

with left_col:
    if page == "Price Prediction":
        st.subheader("Price Prediction Inputs")
        with st.form("price_form"):
            payload = build_listing_inputs("price")
            submit = st.form_submit_button("Predict Price", use_container_width=True)

    elif page == "Revenue Prediction":
        st.subheader("Revenue Prediction Inputs")
        with st.form("revenue_form"):
            payload = build_listing_inputs("revenue")
            submit = st.form_submit_button("Predict Revenue", use_container_width=True)

    else:
        st.subheader("Competitive Analysis Inputs")
        with st.form("competitive_form"):
            payload = build_listing_inputs("competitive")
            submit = st.form_submit_button("Analyze Competitiveness", use_container_width=True)

with right_col:
    st.subheader("Results")
    if "submit" in locals() and submit:
        endpoint = API_ENDPOINTS["price"]
        if page == "Revenue Prediction":
            endpoint = API_ENDPOINTS["revenue"]
        elif page == "Competitive Analysis":
            endpoint = API_ENDPOINTS["competitive"]

        with st.spinner("Processing..."):
            result = call_api(endpoint, payload)

        if not result["success"]:
            st.markdown(
                f'<div class="result-err"><strong>Request failed.</strong><br>{result.get("error", "Unknown error")}</div>',
                unsafe_allow_html=True,
            )
        else:
            data = result["data"]
            if page in {"Price Prediction", "Revenue Prediction"}:
                prediction = float(data.get("prediction", 0.0))
                st.markdown(
                    f'<div class="result-ok"><strong>{data.get("prediction_formatted", "N/A")}</strong><br>'
                    f'Model: {data.get("model", "Unknown")} | Target: {data.get("target", "N/A")}</div>',
                    unsafe_allow_html=True,
                )
                chart_title = "Suggested Nightly Rate" if page == "Price Prediction" else "Projected Monthly Revenue"
                st.plotly_chart(build_price_gauge(prediction, chart_title), use_container_width=True)
            else:
                st.markdown(
                    f'<div class="result-ok"><strong>Market Position: {data.get("market_position", "N/A")}</strong><br>'
                    f'Competitiveness Score: {data.get("competitiveness_score", 0):.1f}</div>',
                    unsafe_allow_html=True,
                )
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Cluster", str(data.get("cluster_id", "-")))
                with c2:
                    st.metric("Cluster Size", str(data.get("cluster_size", "-")))
                with c3:
                    st.metric("Price vs Cluster", f"{data.get('price_vs_cluster', 0):.2f}x")

            with st.expander("Raw API Response", expanded=False):
                st.json(data)
    else:
        st.info("Fill inputs and submit to see results.")

st.divider()
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
