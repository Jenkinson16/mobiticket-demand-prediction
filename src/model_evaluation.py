# src/model_evaluation.py

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import argparse
from pathlib import Path

def evaluate_model(model_path: str, test_data_dir: str):
    """
    Loads a trained model and test data, makes predictions, and prints
    evaluation metrics.

    Args:
        model_path (str): The file path for the trained model pipeline.
        test_data_dir (str): The directory containing X_test.csv and y_test.csv.
    """
    print("Starting model evaluation...")
    
    # Load the model and test data
    model = joblib.load(model_path)
    test_data_path = Path(test_data_dir)
    X_test = pd.read_csv(test_data_path / 'X_test.csv')
    y_test = pd.read_csv(test_data_path / 'y_test.csv').squeeze() # Use squeeze to make it a Series
    
    print(f"Loaded model from {model_path} and test data.")
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Print the results
    print("\n--- Model Evaluation Results ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R-squared (R2 Score): {r2:.2f}")
    print("--------------------------------\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate the demand prediction model.")
    parser.add_argument('--model', type=str, default='models/demand_predictor.pkl', help='Path to the trained model file.')
    parser.add_argument('--data-dir', type=str, default='data/', help='Directory containing the test data (X_test.csv, y_test.csv).')
    
    args = parser.parse_args()
    
    evaluate_model(model_path=args.model, test_data_dir=args.data_dir)