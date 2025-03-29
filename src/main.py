import sys
import os
# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split

from src.data.load_data import load_dataset
from src.data.preprocess import preprocess_data
from src.models.train import train_model, evaluate_model, save_model
from src.visualization.visualize import plot_feature_importance, plot_confusion_matrix

def main(train_data_path, test_data_path=None, output_dir="models", experiment_name="churn_prediction"):
    """
    Main function to run the workflow
    
    Args:
        train_data_path: Path to the training dataset
        test_data_path: Path to the test dataset (optional)
        output_dir: Directory to save outputs
        experiment_name: MLflow experiment name
    """
    # Set up MLflow experiment
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run():
        print("Loading data...")
        df = load_dataset(train_data_path)
        
        # If test path is not provided, split the data
        if test_data_path is None:
            print("Splitting data into train and test sets...")
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        else:
            print("Loading test data...")
            train_df = df
            test_df = load_dataset(test_data_path)
        
        print("Preprocessing data...")
        X_train, y_train, feature_names, encoders = preprocess_data(train_df)
        X_test, y_test, _, _ = preprocess_data(test_df, target_column='Churn')
        
        # Log dataset information
        mlflow.log_param("train_shape", X_train.shape)
        mlflow.log_param("test_shape", X_test.shape)
        
        # Define hyperparameters
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'random_state': 42
        }
        
        print("Training model...")
        model = train_model(X_train, y_train, params=params)
        
        print("Evaluating model...")
        metrics = evaluate_model(model, X_test, y_test)
        
        # Visualizations
        print("Creating visualizations...")
        plot_feature_importance(model, feature_names, save_dir=output_dir)
        y_pred = model.predict(X_test)
        plot_confusion_matrix(y_test, y_pred, save_dir=output_dir)
        
        # Save model and artifacts
        print("Saving model and artifacts...")
        artifact_paths = save_model(model, encoders, feature_names, model_dir=output_dir)
        
        # Log artifacts to MLflow
        for artifact_path in artifact_paths.values():
            mlflow.log_artifact(artifact_path)
        
        # Log model to MLflow
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        print(f"Workflow completed successfully. Model saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a Random Forest model for churn prediction')
    parser.add_argument('--train-data', type=str, required=True, help='Path to training data')
    parser.add_argument('--test-data', type=str, default=None, help='Path to test data (optional)')
    parser.add_argument('--output-dir', type=str, default='models', help='Directory to save outputs')
    parser.add_argument('--experiment-name', type=str, default='churn_prediction', help='MLflow experiment name')
    
    args = parser.parse_args()
    
    main(args.train_data, args.test_data, args.output_dir, args.experiment_name)