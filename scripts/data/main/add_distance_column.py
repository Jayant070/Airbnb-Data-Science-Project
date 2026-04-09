import os
import sys
import pandas as pd
from pathlib import Path

# Add geoencoding script to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "geoencoding"))
from geo_encoding import resolve_city_distance

# Anomaly detection threshold (km)
# Distances greater than this are flagged as potentially erroneous
MAX_REASONABLE_DISTANCE = 100  # km

# Minimum population for nearest city matching
# Increased from default to filter out very small places, but set low enough
# to include legitimate cities. Langebaan (9940) should be included.
MIN_POPULATION_FOR_NEAREST = 5000


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
    
    # Define paths
    merged_csv_path = project_root / "data" / "processed" / "main" / "merged.csv"
    cities_csv_path = project_root / "data" / "processed" / "external" / "cities.csv"
    alias_map_path = project_root / "data" / "processed" / "external" / "alias_map.pkl"
    output_csv_path = project_root / "data" / "processed" / "main" / "merged.csv"
    
    # Validate input file exists
    if not merged_csv_path.exists():
        print(f"Error: {merged_csv_path} not found")
        return False
    
    # Validate cities data exists
    if not cities_csv_path.exists():
        print(f"Error: {cities_csv_path} not found")
        return False
    
    if not alias_map_path.exists():
        print(f"Warning: {alias_map_path} not found (will proceed with empty alias map)")
    
    print(f"Loading merged dataset from: {merged_csv_path}")
    df = pd.read_csv(merged_csv_path)
    
    # Validate required columns
    required_columns = {"latitude", "longitude"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        return False
    
    print(f"Dataset shape: {df.shape}")
    print(f"Processing {len(df)} listings...")
    
    distances = []
    city_populations = []
    errors = []
    anomalies = []  # Track anomalous distances
    
    # Process each row
    for idx, row in df.iterrows():
        try:
            city = row.get("city")
            latitude = row.get("latitude")
            longitude = row.get("longitude")
            country_code = row.get("country_code") if "country_code" in row else None

            # Skip if coordinates are missing
            if pd.isna(latitude) or pd.isna(longitude):
                distances.append(None)
                city_populations.append(None)
                if idx % 100 == 0:
                    print(f"Warning: Row {idx + 1}: Missing coordinates, skipping")
                continue

            # Resolve using coordinates only (no city name verification)
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
            
            # Check for anomalies (unreasonably large distances)
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
                print(f"Warning: Row {idx + 1}: ANOMALY - Distance {distance_km:.2f} km exceeds threshold ({MAX_REASONABLE_DISTANCE} km)")
                distances.append(distance_km)  # Still add it, but flag as anomalous
                city_populations.append(city_population)
                continue
            
            # If anomaly, skip printing the row and continue (already appended and logged above)
            if not (distance_km is not None and distance_km > MAX_REASONABLE_DISTANCE):
                if idx % 100 == 0:
                    print(f"Row {idx + 1}: ({latitude},{longitude}) -> Distance: {distance_km:.2f} km | population: {city_population}")

            distances.append(distance_km)
            city_populations.append(city_population)
        
        except Exception as e:
            print(f"Error: Row {idx + 1}: Error processing {row.get('city')} - {str(e)}")
            distances.append(None)
            city_populations.append(None)
            errors.append((idx, str(e)))
    
    # Add the new column
    df["distance_from_city_center"] = distances
    df["city_population"] = city_populations
    
    # Save the updated dataset
    print(f"\nSaving updated dataset to: {output_csv_path}")
    df.to_csv(output_csv_path, index=False)
    
    # Handle anomalies: save to file for review
    if anomalies:
        anomalies_df = pd.DataFrame(anomalies)
        anomalies_csv_path = project_root / "data" / "processed" / "main" / "anomalies_report.csv"
        print(f"\nWarning: Saving {len(anomalies)} anomalous records to: {anomalies_csv_path}")
        anomalies_df.to_csv(anomalies_csv_path, index=False)
    
    # Print summary
    print("\nSummary:")
    print(f"   Total rows processed: {len(df)}")
    print(f"   Rows with valid distances: {sum(1 for d in distances if d is not None)}")
    print(f"   Rows with missing distances: {sum(1 for d in distances if d is None)}")
    print(f"   Rows with ANOMALIES (distance > {MAX_REASONABLE_DISTANCE} km): {len(anomalies)}")
    print(f"   Errors encountered: {len(errors)}")
    
    # Display anomalies
    if anomalies:
        print("\nAnomalous records details")
        print(f"\nTotal Anomalies Found: {len(anomalies)}\n")
        for i, anomaly in enumerate(anomalies, 1):
            print(f"{i}. Row #{anomaly['row_index']}")
            print(f"   City Name: {anomaly['city']}")
            print(f"   Coordinates: ({anomaly['latitude']}, {anomaly['longitude']})")
            print(f"   Distance Found: {anomaly['distance_km']:.2f} km (Threshold: {MAX_REASONABLE_DISTANCE} km)")
            print(f"   Match Method: {anomaly['match_method']}")
            print(f"   Matched to: {anomaly['matched_city']}")
            print(f"   Matched Coordinates: ({anomaly['matched_latitude']}, {anomaly['matched_longitude']})")
            print()
        print("Anomalous records saved to: anomalies_report.csv\n")
        print("Suggestions for investigation:")
        print(f"   1. Review 'anomalies_report.csv' for detailed records")
        print(f"   2. Check if city names are spelled correctly")
        print(f"   3. Verify latitude/longitude are in correct range (-90 to 90 / -180 to 180)")
        print(f"   4. Check for data entry errors or typos")
        print(f"   5. If data is valid, increase MAX_REASONABLE_DISTANCE threshold (currently {MAX_REASONABLE_DISTANCE} km)")
    
    if errors:
        print(f"\nWarning: Processing errors (first 5):")
        for idx, error in errors[:5]:
            print(f"   Row {idx + 1}: {error}")
    
    print("\nSuccessfully added 'distance_from_city_center' column!")
    if distances:
        valid_distances = [d for d in distances if d is not None]
        if valid_distances:
            print(f"   Min distance: {min(valid_distances):.2f} km")
            print(f"   Max distance: {max(valid_distances):.2f} km")
            print(f"   Avg distance: {sum(valid_distances) / len(valid_distances):.2f} km")
    print(f"   Anomaly threshold (MAX_REASONABLE_DISTANCE): {MAX_REASONABLE_DISTANCE} km")
    print(f"   Population threshold (MIN_POPULATION_FOR_NEAREST): {MIN_POPULATION_FOR_NEAREST}")
    
    return True


if __name__ == "__main__":
    success = add_distance_column()
    sys.exit(0 if success else 1)
