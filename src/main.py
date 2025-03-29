import sys
import os
# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from datetime import datetime
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
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up MLflow experiment
    mlflow.set_experiment(experiment_name)
    
    # Set up run name with timestamp for better tracking
    run_name = f"churn_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Define hyperparameters
    params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 'auto',
        'bootstrap': True,
        'random_state': 42
    }
    
    with mlflow.start_run(run_name=run_name):
        # Set tags for better organization
        mlflow.set_tag("model_type", "random_forest")
        mlflow.set_tag("data_source", os.path.basename(train_data_path))
        mlflow.set_tag("purpose", "churn_prediction")
        
        print("Loading data...")
        df = load_dataset(train_data_path)
        
        # Log dataset parameters
        mlflow.log_param("dataset_rows", len(df))
        mlflow.log_param("dataset_columns", df.shape[1])
        
        # If test path is not provided, split the data
        if test_data_path is None:
            print("Splitting data into train and test sets...")
            mlflow.log_param("split_type", "train_test_split")
            mlflow.log_param("test_size", 0.2)
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        else:
            print("Loading test data...")
            mlflow.log_param("split_type", "separate_files")
            train_df = df
            test_df = load_dataset(test_data_path)
        
        print("Preprocessing data...")
        X_train, y_train, feature_names, encoders = preprocess_data(train_df)
        X_test, y_test, _, _ = preprocess_data(test_df, target_column='Churn')
        
        # Log dataset information
        mlflow.log_param("train_samples", X_train.shape[0])
        mlflow.log_param("test_samples", X_test.shape[0])
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("feature_names", ", ".join(feature_names))
        
        # Log preprocessing info
        mlflow.log_param("categorical_features", len([e for e in encoders if e != 'scaler']))
        
        # Log all hyperparameters
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)
        
        print("Training model...")
        model = train_model(X_train, y_train, params=params)
        
        print("Evaluating model...")
        metrics = evaluate_model(model, X_test, y_test)
        
        # Log all metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Visualizations
        print("Creating visualizations...")
        feature_importance_path = plot_feature_importance(model, feature_names, save_dir=output_dir)
        mlflow.log_artifact(feature_importance_path, "visualizations")
        
        y_pred = model.predict(X_test)
        conf_matrix_path = plot_confusion_matrix(y_test, y_pred, save_dir=output_dir)
        mlflow.log_artifact(conf_matrix_path, "visualizations")
        
        # Save model and artifacts locally
        print("Saving model and artifacts...")
        artifact_paths = save_model(model, encoders, feature_names, model_dir=output_dir)
        
        # Log artifacts to MLflow
        for artifact_name, artifact_path in artifact_paths.items():
            mlflow.log_artifact(artifact_path, artifact_name)
        
        # Log model to MLflow with signature
        feature_spec = mlflow.models.infer_signature(X_train, y_train)
        mlflow.sklearn.log_model(
            model, 
            "random_forest_model", 
            signature=feature_spec,
            input_example=X_train[:5]
        )
        
        # Create a summary of the run
        run_id = mlflow.active_run().info.run_id
        print(f"Workflow completed successfully. Model saved to {output_dir}")
        print(f"MLflow run ID: {run_id}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        
        # Return metrics for potential further use
        return metrics, run_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a Random Forest model for churn prediction')
    parser.add_argument('--train-data', type=str, required=True, help='Path to training data')
    parser.add_argument('--test-data', type=str, default=None, help='Path to test data (optional)')
    parser.add_argument('--output-dir', type=str, default='models', help='Directory to save outputs')
    parser.add_argument('--experiment-name', type=str, default='churn_prediction', help='MLflow experiment name')
    
    args = parser.parse_args()
    
    main(args.train_data, args.test_data, args.output_dir, args.experiment_name)
