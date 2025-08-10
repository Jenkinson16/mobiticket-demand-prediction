# src/data_preprocessing.py

import pandas as pd
import argparse
from pathlib import Path

def preprocess_data(raw_data_path: str, output_path: str):
    """
    Reads raw transactional data, aggregates it per ride, engineers features,
    and saves the processed data to a new CSV file.

    Args:
        raw_data_path (str): The file path for the raw CSV data.
        output_path (str): The file path to save the processed CSV data.
    """
    print("Starting data preprocessing...")

    # Define output directory and create if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = pd.read_csv(raw_data_path)
    print(f"Loaded raw data with shape: {df.shape}")

    # Aggregate data by ride_id to get one row per ride
    ride_df = df.groupby('ride_id').agg(
        travel_date=('travel_date', 'first'),
        travel_time=('travel_time', 'first'),
        travel_from=('travel_from', 'first'),
        car_type=('car_type', 'first'),
        max_capacity=('max_capacity', 'first'),
        number_of_seats=('ride_id', 'count')
    ).reset_index()
    print(f"Aggregated data to ride level. New shape: {ride_df.shape}")

    # --- Feature Engineering ---
    # Convert date and time columns to datetime objects
    ride_df['travel_date'] = pd.to_datetime(ride_df['travel_date'])
    
    # Extract features from travel_date
    ride_df['day_of_week'] = ride_df['travel_date'].dt.dayofweek
    ride_df['day_of_year'] = ride_df['travel_date'].dt.dayofyear
    ride_df['month'] = ride_df['travel_date'].dt.month
    ride_df['year'] = ride_df['travel_date'].dt.year
    ride_df['is_weekend'] = ride_df['day_of_week'].isin([5, 6]).astype(int)

    # Extract hour from travel_time
    ride_df['departure_hour'] = pd.to_datetime(ride_df['travel_time'], format='%H:%M').dt.hour

    # Drop original and unnecessary columns
    ride_df = ride_df.drop(columns=['travel_date', 'travel_time', 'ride_id'])
    
    print("Feature engineering complete.")
    
    # Save the processed data
    ride_df.to_csv(output_path, index=False)
    print(f"Processed data saved successfully to {output_path}")


if __name__ == '__main__':
    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(description="Preprocess Mobiticket raw data.")
    parser.add_argument('--input', type=str, default='data/train_revised.csv', help='Path to the raw input CSV file.')
    parser.add_argument('--output', type=str, default='data/processed_data.csv', help='Path to save the processed output CSV file.')
    
    args = parser.parse_args()
    
    preprocess_data(raw_data_path=args.input, output_path=args.output)