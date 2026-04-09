"""
Merge multiple CSV files from raw dataset into a single file.

This script scans the data/raw/main directory for CSV files, validates they have
consistent schemas, and merges them into a single CSV file saved in
data/processed/main.
"""

import os
import sys
import pandas as pd
from pathlib import Path


def find_csv_files(root_dir):
    """Find all CSV files in subfolders of root_dir, skipping any in root_dir itself."""
    csv_files = []
    for entry in os.scandir(root_dir):
        if entry.is_dir():
            for root, dirs, files in os.walk(entry.path):
                for file in files:
                    if file.endswith(".csv"):
                        csv_files.append(os.path.join(root, file))
    return csv_files


def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent.parent


def merge_csv_files(root_dir, output_file):
    """
    Merge all CSV files from root_dir into a single output file.
    
    Validates that all CSVs have the same columns before merging.
    """
    csv_files = find_csv_files(root_dir)

    if not csv_files:
        print("Error: No CSV files found.")
        return False

    print(f"Found {len(csv_files)} CSV files in {root_dir}")

    # Read first file as reference
    try:
        base_df = pd.read_csv(csv_files[0])
        base_columns = set(base_df.columns)
    except Exception as e:
        print(f"Error reading {csv_files[0]}: {e}")
        return False

    print(f"Base schema: {sorted(base_columns)}")

    all_dfs = [base_df]
    skipped_files = []

    # Check remaining files
    for file in csv_files[1:]:
        try:
            df = pd.read_csv(file)
            current_columns = set(df.columns)

            if current_columns != base_columns:
                print(f"\nWarning: Column mismatch detected in: {file}")

                missing = base_columns - current_columns
                extra = current_columns - base_columns

                if missing:
                    print(f"   Missing columns: {missing}")
                if extra:
                    print(f"   Extra columns: {extra}")

                skipped_files.append(file)
                continue

            # Ensure same column order
            df = df[sorted(base_columns)]
            all_dfs.append(df)
            print(f"Added: {file}")

        except Exception as e:
            print(f"Error reading {file}: {e}")
            skipped_files.append(file)

    # Merge all
    merged_df = pd.concat(all_dfs, ignore_index=True)

    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        merged_df.to_csv(output_file, index=False)
        print(f"\nSuccessfully merged {len(all_dfs)} files into:")
        print(f"   {output_file}")
        print(f"Output shape: {merged_df.shape}")
        
        if skipped_files:
            print(f"\nSkipped files ({len(skipped_files)}):")
            for f in skipped_files:
                print(f"   {f}")
        
        return True
    except Exception as e:
        print(f"Error saving merged file: {e}")
        return False


if __name__ == "__main__":
    project_root = get_project_root()
    
    # Input directory: raw data
    input_directory = project_root / "data" / "raw" / "main"
    
    # Output file: processed data
    output_directory = project_root / "data" / "processed" / "main"
    output_file = output_directory / "merged.csv"
    
    success = merge_csv_files(str(input_directory), str(output_file))
    sys.exit(0 if success else 1)