import os
import sys
import logging
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "geoencoding"))
from geo_encoding import resolve_city_distance

MAX_REASONABLE_DISTANCE = 100  # km

MIN_POPULATION_FOR_NEAREST = 5000

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent.parent


def add_distance_column():
    """
    Load merged.csv, calculate distances, and add distance_from_city_center column.
    
    Now continues processing all rows and records anomalies instead of stopping.
    Anomalous records are saved to a separate file for review.
    """
    project_root = get_project_root()

    merged_csv_path = project_root / "data" / "processed" / "main" / "merged.csv"
    cities_csv_path = project_root / "data" / "processed" / "external" / "cities.csv"
    alias_map_path = project_root / "data" / "processed" / "external" / "alias_map.pkl"
    output_csv_path = project_root / "data" / "processed" / "main" / "merged.csv"

    if not merged_csv_path.exists():
        logger.error(f"{merged_csv_path} not found")
        return False

    if not cities_csv_path.exists():
        logger.error(f"{cities_csv_path} not found")
        return False

    if not alias_map_path.exists():
        logger.warning(f"{alias_map_path} not found; proceeding with empty alias map")

    logger.info(f"Loading merged dataset from: {merged_csv_path}")
    df = pd.read_csv(merged_csv_path)

    required_columns = {"latitude", "longitude"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return False

    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Processing {len(df)} listings")

    distances = []
    city_populations = []
    errors = []
    anomalies = []

    for idx, row in df.iterrows():
        try:
            latitude = row.get("latitude")
            longitude = row.get("longitude")

            if pd.isna(latitude) or pd.isna(longitude):
                distances.append(None)
                city_populations.append(None)
                continue

            result = resolve_city_distance(
                cityname=None,
                latitude=float(latitude),
                longitude=float(longitude),
                country_code=None,
                csv_path=str(cities_csv_path),
                alias_map_path=str(alias_map_path),
                min_population=MIN_POPULATION_FOR_NEAREST,
            )

            distance_km = result.get("distance_from_city_center", result.get("distance_km", None))
            city_population = result.get("city_population", result.get("population", None))

            if distance_km is not None and distance_km > MAX_REASONABLE_DISTANCE:
                anomaly_record = {
                    "row_index": idx + 1,
                    "latitude": latitude,
                    "longitude": longitude,
                    "distance_km": distance_km,
                    "match_method": result.get("match_method", "unknown"),
                    "matched_city": result.get("name", "unknown"),
                    "matched_latitude": result.get("latitude", None),
                    "matched_longitude": result.get("longitude", None),
                }
                anomalies.append(anomaly_record)
                logger.warning(
                    f"Row {idx + 1}: distance {distance_km:.2f} km exceeds threshold {MAX_REASONABLE_DISTANCE} km"
                )
                distances.append(distance_km)
                city_populations.append(city_population)
                continue

            distances.append(distance_km)
            city_populations.append(city_population)
        
        except Exception as e:
            logger.error(f"Row {idx + 1}: Error processing record - {str(e)}")
            distances.append(None)
            city_populations.append(None)
            errors.append((idx, str(e)))

    df["distance_from_city_center"] = distances
    df["city_population"] = city_populations

    logger.info(f"Saving updated dataset to: {output_csv_path}")
    df.to_csv(output_csv_path, index=False)

    if anomalies:
        anomalies_df = pd.DataFrame(anomalies)
        anomalies_csv_path = project_root / "data" / "processed" / "main" / "anomalies_report.csv"
        logger.warning(f"Saving {len(anomalies)} anomalous records to: {anomalies_csv_path}")
        anomalies_df.to_csv(anomalies_csv_path, index=False)

    logger.info(f"Total rows processed: {len(df)}")
    logger.info(f"Rows with valid distances: {sum(1 for d in distances if d is not None)}")
    logger.info(f"Rows with missing distances: {sum(1 for d in distances if d is None)}")
    logger.info(f"Rows with anomalies: {len(anomalies)}")
    logger.info(f"Errors encountered: {len(errors)}")

    if errors:
        logger.warning("Processing errors (first 5):")
        for idx, error in errors[:5]:
            logger.warning(f"Row {idx + 1}: {error}")

    logger.info("Successfully added 'distance_from_city_center' column")
    if distances:
        valid_distances = [d for d in distances if d is not None]
        if valid_distances:
            logger.info(f"Min distance: {min(valid_distances):.2f} km")
            logger.info(f"Max distance: {max(valid_distances):.2f} km")
            logger.info(f"Avg distance: {sum(valid_distances) / len(valid_distances):.2f} km")
    logger.info(f"Anomaly threshold: {MAX_REASONABLE_DISTANCE} km")
    logger.info(f"Population threshold: {MIN_POPULATION_FOR_NEAREST}")

    return True


if __name__ == "__main__":
    success = add_distance_column()
    sys.exit(0 if success else 1)
