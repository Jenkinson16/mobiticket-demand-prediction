# src/model_training.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import joblib
import argparse
from pathlib import Path

def train_model(processed_data_path: str, model_output_path: str):
    """
    Loads processed data, trains a RandomForestRegressor model, and saves the
    trained pipeline.

    Args:
        processed_data_path (str): The file path for the processed data.
        model_output_path (str): The file path to save the trained model.
    """
    print("Starting model training...")

    # Define output directory and create if it doesn't exist
    output_dir = Path(model_output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(processed_data_path)
    print("Loaded processed data.")

    # Define features (X) and target (y)
    X = df.drop('number_of_seats', axis=1)
    y = df['number_of_seats']

    # Identify categorical and numerical features
    categorical_features = ['travel_from', 'car_type']
    numerical_features = X.columns.drop(categorical_features).tolist()

    # Create the preprocessor object using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    # Define the model pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data split into training ({X_train.shape[0]} rows) and testing ({X_test.shape[0]} rows).")

    # Train the model
    model_pipeline.fit(X_train, y_train)
    print("Model training complete.")

    # Save the trained model pipeline
    joblib.dump(model_pipeline, model_output_path)
    print(f"Model pipeline saved successfully to {model_output_path}")

    # Save the test set for evaluation
    test_set_dir = Path(processed_data_path).parent
    X_test.to_csv(test_set_dir / 'X_test.csv', index=False)
    y_test.to_csv(test_set_dir / 'y_test.csv', index=False)
    print(f"Test data saved for evaluation.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a demand prediction model.")
    parser.add_argument('--input', type=str, default='data/processed_data.csv', help='Path to the processed input CSV file.')
    parser.add_argument('--output', type=str, default='models/demand_predictor.pkl', help='Path to save the trained model.')
    
    args = parser.parse_args()
    
    train_model(processed_data_path=args.input, model_output_path=args.output)