import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(df, target_column='Churn'):
    """
    Preprocess the data for model training
    
    Args:
        df: Input DataFrame
        target_column: Target variable column name
    
    Returns:
        X: Features DataFrame
        y: Target variable
        feature_names: List of feature names
        encoders: Dictionary of encoders used
    """
    # Make a copy to avoid modifying the original
    data = df.copy()
    
    # Handle missing values
    data = data.fillna(data.median(numeric_only=True))
    
    # Extract target variable
    if target_column in data.columns:
        y = data[target_column]
        X = data.drop(target_column, axis=1)
    else:
        raise ValueError(f"Target column '{target_column}' not found in dataset")
    
    # Initialize dictionary to store encoders
    encoders = {}
    
    # Process categorical features
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le
    
    # Scale numeric features
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    encoders['scaler'] = scaler
    
    feature_names = X.columns.tolist()
    
    return X, y, feature_names, encoders