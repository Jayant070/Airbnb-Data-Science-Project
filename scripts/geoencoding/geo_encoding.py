"""City lookup and distance utilities.

The main entry point is ``resolve_city_distance``.
It prefers a normalized city name when one is provided, otherwise it
finds the nearest city to the supplied coordinates.
"""

from __future__ import annotations

import math
import os
import pickle
import argparse
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0

def _validate_coordinates(latitude: float, longitude: float) -> None:
	"""Validate latitude and longitude ranges."""

	if not (-90 <= float(latitude) <= 90):
		raise ValueError(f"latitude must be between -90 and 90, got {latitude}")
	if not (-180 <= float(longitude) <= 180):
		raise ValueError(f"longitude must be between -180 and 180, got {longitude}")


@lru_cache(maxsize=4)
def load_cities(csv_path: str = "cities.csv") -> pd.DataFrame:
	"""Load the city dataset."""

	# Only use the built-in default path when the default value is used.
	# User-provided relative paths should resolve from the current working directory.
	if not csv_path or csv_path == "cities.csv":
		script_dir = os.path.dirname(os.path.abspath(__file__))
		csv_path = os.path.join(script_dir, "..", "..", "data", "processed", "external", "cities.csv")
	elif not os.path.isabs(csv_path):
		csv_path = os.path.abspath(csv_path)

	df = pd.read_csv(csv_path)
	required_columns = {"name", "latitude", "longitude", "country_code"}
	missing_columns = required_columns - set(df.columns)
	if missing_columns:
		if {"city", "country", "country_code"}.issubset(set(df.columns)):
			raise ValueError(
				f"CSV at {csv_path} appears to be a listing-level city file, not the geocoding cities master. "
				"Expected columns: name, latitude, longitude, country_code. "
				"Tip: use data/processed/external/cities.csv or pass --csv with the correct file."
			)
		raise ValueError(
			f"Missing required columns in {csv_path}: {sorted(missing_columns)}. "
			f"Found columns: {list(df.columns)}"
		)

	df = df.copy()
	df["name"] = df["name"].astype(str)
	df["country_code"] = df["country_code"].astype(str).str.upper().str.strip()
	df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
	df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
	df = df.dropna(subset=["latitude", "longitude", "name", "country_code"])
	return df.reset_index(drop=True)


@lru_cache(maxsize=4)
def load_alias_map(pkl_path: str = "alias_map.pkl") -> Dict[str, str]:
	"""Load the optional alias map.

	If the file does not exist, an empty mapping is returned.
	"""

	if not pkl_path or pkl_path == "alias_map.pkl":
		script_dir = os.path.dirname(os.path.abspath(__file__))
		pkl_path = os.path.join(script_dir, "..", "..", "data", "processed", "external", "alias_map.pkl")
	elif not os.path.isabs(pkl_path):
		pkl_path = os.path.abspath(pkl_path)

	if not os.path.exists(pkl_path):
		return {}

	with open(pkl_path, "rb") as handle:
		alias_map = pickle.load(handle)

	if not isinstance(alias_map, dict):
		raise ValueError("alias_map.pkl must contain a dictionary")

	normalized_map: Dict[str, str] = {}
	for key, value in alias_map.items():
		if key is None or value is None:
			continue
		normalized_map[str(key).lower().strip()] = str(value).strip()
	return normalized_map


def normalize_city(name: str, alias_map: Dict[str, str]) -> str:
	"""Normalize a city name through the alias map."""

	if not name:
		return ""
	return alias_map.get(name.lower().strip(), "")


def haversine_distance(
	latitude_1: float,
	longitude_1: float,
	latitude_2: float,
	longitude_2: float,
) -> float:
	"""Return the great-circle distance between two points in kilometers."""

	latitude_1 = math.radians(float(latitude_1))
	longitude_1 = math.radians(float(longitude_1))
	latitude_2 = math.radians(float(latitude_2))
	longitude_2 = math.radians(float(longitude_2))

	delta_lat = latitude_2 - latitude_1
	delta_lon = longitude_2 - longitude_1
	a = (
		math.sin(delta_lat / 2) ** 2
		+ math.cos(latitude_1) * math.cos(latitude_2) * math.sin(delta_lon / 2) ** 2
	)
	c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
	return EARTH_RADIUS_KM * c


def find_city_by_name(
	cities_df: pd.DataFrame,
	city_name: str,
	country_code: Optional[str] = None,
	latitude: Optional[float] = None,
	longitude: Optional[float] = None,
) -> Optional[pd.Series]:
	"""Return the best matching city row for a canonical city name.

	If multiple rows match the same canonical name (for example, duplicate city names
	in the same country), the closest row to the provided coordinates is returned.
	"""

	if not city_name:
		return None

	normalized_name = city_name.lower().strip()
	matches = cities_df[cities_df["name"].astype(str).str.lower().str.strip() == normalized_name]
	if country_code:
		normalized_country_code = str(country_code).upper().strip()
		matches = matches[matches["country_code"] == normalized_country_code]
	if matches.empty:
		return None

	if latitude is None or longitude is None or len(matches) == 1:
		return matches.iloc[0]

	closest_match = None
	closest_distance = float("inf")
	for _, candidate in matches.iterrows():
		distance_km = haversine_distance(
			latitude,
			longitude,
			candidate["latitude"],
			candidate["longitude"],
		)
		if distance_km < closest_distance:
			closest_distance = distance_km
			closest_match = candidate

	return closest_match


def find_nearest_city(
	cities_df: pd.DataFrame,
	latitude: float,
	longitude: float,
	country_code: Optional[str] = None,
	min_population: int = 0,
) -> Tuple[pd.Series, float]:
	"""Find the nearest city to the input coordinates, optionally constrained by country."""

	candidate_df = cities_df.copy()
	
	if country_code:
		normalized_country_code = str(country_code).upper().strip()
		candidate_df = candidate_df[candidate_df["country_code"] == normalized_country_code]
		if candidate_df.empty:
			raise ValueError(f"No cities found for country_code={normalized_country_code}")

	if "population" in candidate_df.columns and min_population > 0:
		# Filter by population, but use fillna(0) to handle missing values
		candidate_df = candidate_df[pd.to_numeric(candidate_df["population"], errors="coerce").fillna(0) >= min_population]
		if candidate_df.empty:
			raise ValueError(f"No cities found with population >= {min_population}")

	latitudes = np.radians(candidate_df["latitude"].to_numpy(dtype=float))
	longitudes = np.radians(candidate_df["longitude"].to_numpy(dtype=float))
	input_latitude = math.radians(float(latitude))
	input_longitude = math.radians(float(longitude))

	delta_lat = latitudes - input_latitude
	delta_lon = longitudes - input_longitude
	a = (
		np.sin(delta_lat / 2) ** 2
		+ np.cos(input_latitude) * np.cos(latitudes) * np.sin(delta_lon / 2) ** 2
	)
	distances = 2 * EARTH_RADIUS_KM * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
	nearest_index = int(np.argmin(distances))
	return candidate_df.iloc[nearest_index], float(distances[nearest_index])


def resolve_city_distance(
	cityname: Optional[str],
	latitude: float,
	longitude: float,
	country_code: Optional[str] = None,
	csv_path: str = None,
	alias_map_path: str = None,
	min_population: int = 0,
	max_name_match_distance_km: float = 50.0,
) -> Dict[str, Any]:
	"""Resolve city metadata from coordinates.

	Uses geographic coordinates to find the nearest city.

	Returns a dictionary with the nearest city record, distance, and city population.
	"""

	if latitude is None or longitude is None:
		raise ValueError("latitude and longitude are required")

	_validate_coordinates(latitude, longitude)

	# Use default paths if not provided
	if not csv_path:
		script_dir = os.path.dirname(os.path.abspath(__file__))
		csv_path = os.path.join(script_dir, "..", "..", "data", "processed", "external", "cities.csv")
	if not alias_map_path:
		script_dir = os.path.dirname(os.path.abspath(__file__))
		alias_map_path = os.path.join(script_dir, "..", "..", "data", "processed", "external", "alias_map.pkl")

	cities_df = load_cities(csv_path)
	
	normalized_country_code = str(country_code).upper().strip() if country_code else ""

	# Find nearest city based on coordinates only
	selected_city, distance_km = find_nearest_city(
		cities_df,
		latitude,
		longitude,
		normalized_country_code,
		min_population=min_population,
	)
	
	match_method = "nearest_country" if normalized_country_code else "nearest"

	result = selected_city.to_dict()
	
	zone = selected_city["zone"] if "zone" in selected_city else None
	city_population = selected_city["population"] if "population" in selected_city else None
	result.update(
		{
			"input_cityname": cityname or "",
			"input_country_code": normalized_country_code,
			"match_method": match_method,
			"distance_km": distance_km,
			"distance_from_city_center": distance_km,
			"city_population": city_population,
			"zone": zone,
		}
	)
	return result


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Resolve nearest city using coordinates only")
	parser.add_argument("--latitude", type=float, required=True, help="Input latitude")
	parser.add_argument("--longitude", type=float, required=True, help="Input longitude")
	parser.add_argument("--country-code", type=str, default=None, help="Optional country code filter")
	parser.add_argument("--csv", type=str, default="cities.csv", help="Path to cities CSV")
	parser.add_argument(
		"--alias-map",
		type=str,
		default="alias_map.pkl",
		help="Path to alias map pickle",
	)
	parser.add_argument("--min-population", type=int, default=0, help="Minimum population for nearest city matching")

	args = parser.parse_args()
	result = resolve_city_distance(
		cityname=None,
		latitude=args.latitude,
		longitude=args.longitude,
		country_code=args.country_code,
		csv_path=args.csv,
		alias_map_path=args.alias_map,
		min_population=args.min_population,
	)
	print(result)
