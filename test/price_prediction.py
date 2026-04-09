import requests

url = "http://localhost:8000/api/predict/price"

payload = {
    "bedrooms": 1,
    "beds": 1,
    "baths": 2,
    "amenities_count": 20,
    "amenities": [
        "tv",
        "wifi",
        "kitchen",
        "air conditioning",
        "pool",
        "gym",
        "hot water kettle",
        "ceiling",
        "free parking on premises"
    ],
    "photos_count": 45,
    "superhost": 0,
    "num_reviews": 48,
    "avg_rating": 4.75,
    "latitude": 18.561279,
    "longitude": 73.826403,
    "cancellation_policy": 7.0,
    "min_nights": 1,
    "cleaning_fee": 0,
    "extra_guest_fee": 0,
    "registration": 1,
    "professional_management": 0,
    "listing_type": "Entire home",
    "room_type": "entire_home",
    "ttm_total_days": 365,
    "ttm_blocked_days": 18
}

headers = {
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)

print("Status Code:", response.status_code)
print("Response JSON:", response.json())