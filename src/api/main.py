import sys
import os
# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import List, Dict, Union

# Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn using Random Forest model",
    version="1.0.0"
)

# Define model directory
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")

# Load model and artifacts
model_path = os.path.join(MODEL_DIR, "model.joblib")
encoders_path = os.path.join(MODEL_DIR, "encoders.joblib")
feature_names_path = os.path.join(MODEL_DIR, "feature_names.joblib")

# Check if model files exist
if not (os.path.exists(model_path) and os.path.exists(encoders_path) and os.path.exists(feature_names_path)):
    raise RuntimeError("Model files not found. Please train the model first.")

# Load model and artifacts
model = joblib.load(model_path)
encoders = joblib.load(encoders_path)
feature_names = joblib.load(feature_names_path)

# Define input data model
class CustomerData(BaseModel):
    State: str
    Account_Length: int
    Area_Code: int
    Phone: str
    Intl_Plan: str
    VMail_Plan: str
    VMail_Message: int
    Day_Mins: float
    Day_Calls: int
    Day_Charge: float
    Eve_Mins: float
    Eve_Calls: int
    Eve_Charge: float
    Night_Mins: float
    Night_Calls: int
    Night_Charge: float
    Intl_Mins: float
    Intl_Calls: int
    Intl_Charge: float
    CustServ_Calls: int

# Define response model
class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: bool
    message: str  # Add this new field
    feature_importance: Dict[str, float]

def preprocess_input(customer_data: CustomerData):
    """Preprocess input data for prediction"""
    # Convert to DataFrame
    data_dict = customer_data.dict()
    
    # Define mapping from API feature names to model feature names
    feature_mapping = {
        "State": "State",
        "Account_Length": "Account length",
        "Area_Code": "Area code",
        "Phone": "Phone",
        "Intl_Plan": "International plan",
        "VMail_Plan": "Voice mail plan",
        "VMail_Message": "Number vmail messages",
        "Day_Mins": "Total day minutes",
        "Day_Calls": "Total day calls",
        "Day_Charge": "Total day charge",
        "Eve_Mins": "Total eve minutes",
        "Eve_Calls": "Total eve calls", 
        "Eve_Charge": "Total eve charge",
        "Night_Mins": "Total night minutes",
        "Night_Calls": "Total night calls",
        "Night_Charge": "Total night charge",
        "Intl_Mins": "Total intl minutes",
        "Intl_Calls": "Total intl calls",
        "Intl_Charge": "Total intl charge",
        "CustServ_Calls": "Customer service calls"
    }
    
    # Rename keys in the data dictionary
    renamed_dict = {feature_mapping.get(k, k): v for k, v in data_dict.items()}
    
    df = pd.DataFrame([renamed_dict])
    
    
    # Preprocess categorical features
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col in encoders:
            # Handle unseen categories
            try:
                df[col] = encoders[col].transform(df[col])
            except ValueError:
                # If category not seen during training, use most frequent category
                print(f"Warning: Unseen category in {col}. Using default value.")
                df[col] = 0  # Use default value
    
    # Scale numerical features
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if 'scaler' in encoders:
        df[numeric_cols] = encoders['scaler'].transform(df[numeric_cols])
    
    # Ensure all feature names are present in the expected order
    input_array = []
    for feature in feature_names:
        if feature in df.columns:
            input_array.append(df[feature].values[0])
        else:
            input_array.append(0)  # Default value for missing features
    
    return np.array(input_array).reshape(1, -1)

@app.get("/")
def read_root():
    return {"message": "Customer Churn Prediction API"}

@app.get("/health")
def health_check():
    """Check if the API is running and model is loaded"""
    if model is None or encoders is None or feature_names is None:
        raise HTTPException(status_code=500, detail="Model or artifacts not loaded correctly")
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer_data: CustomerData):
    """Predict customer churn probability"""
    try:
        # Preprocess input data
        input_data = preprocess_input(customer_data)
        
        # Make prediction
        churn_probability = model.predict_proba(input_data)[0, 1]
        churn_prediction = bool(churn_probability >= 0.5)
        
        # Create the human-readable message
        message = "The customer will churn." if churn_prediction else "The customer will not churn."
        
        # Get feature importances
        importance_dict = {}
        if hasattr(model, 'feature_importances_'):
            for i, feature in enumerate(feature_names):
                importance_dict[feature] = float(model.feature_importances_[i])
        
        # Return prediction
        return PredictionResponse(
            churn_probability=float(churn_probability),
            churn_prediction=churn_prediction,
            message=message,  # Add the message here
            feature_importance=importance_dict
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/features")
def get_features():
    """Return the list of features used by the model"""
    return {"features": feature_names}