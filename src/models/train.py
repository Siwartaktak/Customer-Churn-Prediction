from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import joblib
import os

def train_model(X_train, y_train, params=None, log_mlflow=True):
    """
    Train a Random Forest model
    
    Args:
        X_train: Training features
        y_train: Training target
        params: Model hyperparameters (dict)
        log_mlflow: Whether to log metrics to MLflow
    
    Returns:
        Trained model
    """
    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'random_state': 42
        }
    
    print(f"Training Random Forest with params: {params}")
    
    # Train the model
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    # Log to MLflow if enabled
    if log_mlflow:
        mlflow.log_params(params)
        # Feature importances
        if hasattr(model, 'feature_importances_'):
            for i, importance in enumerate(model.feature_importances_):
                mlflow.log_metric(f"feature_importance_{i}", importance)
    
    return model

def evaluate_model(model, X_test, y_test, log_mlflow=True):
    """
    Evaluate a trained model
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        log_mlflow: Whether to log metrics to MLflow
    
    Returns:
        Dictionary of evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    print("Model Evaluation Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    
    # Log to MLflow if enabled
    if log_mlflow:
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
    
    return metrics

def save_model(model, encoders, feature_names, model_dir="models"):
    """
    Save the model and related artifacts
    
    Args:
        model: Trained model
        encoders: Dictionary of encoders used in preprocessing
        feature_names: List of feature names
        model_dir: Directory to save the model
    
    Returns:
        Dictionary with paths to saved artifacts
    """
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the model
    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(model, model_path)
    
    # Save encoders
    encoders_path = os.path.join(model_dir, "encoders.joblib")
    joblib.dump(encoders, encoders_path)
    
    # Save feature names
    feature_names_path = os.path.join(model_dir, "feature_names.joblib")
    joblib.dump(feature_names, feature_names_path)
    
    print(f"Model and artifacts saved to {model_dir}")
    
    return {
        "model_path": model_path,
        "encoders_path": encoders_path,
        "feature_names_path": feature_names_path
    }