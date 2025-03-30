from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import os
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

# Define input data model with Field aliases for spaces in field names
class CustomerData(BaseModel):
    State: str
    account_length: int = Field(..., alias="Account length")
    area_code: int = Field(..., alias="Area code")
    Phone: str
    international_plan: str = Field(..., alias="International plan")
    voice_mail_plan: str = Field(..., alias="Voice mail plan")
    vmail_messages: int = Field(..., alias="Number vmail messages")
    day_minutes: float = Field(..., alias="Total day minutes")
    day_calls: int = Field(..., alias="Total day calls")
    day_charge: float = Field(..., alias="Total day charge")
    eve_minutes: float = Field(..., alias="Total eve minutes")
    eve_calls: int = Field(..., alias="Total eve calls")
    eve_charge: float = Field(..., alias="Total eve charge")
    night_minutes: float = Field(..., alias="Total night minutes")
    night_calls: int = Field(..., alias="Total night calls")
    night_charge: float = Field(..., alias="Total night charge")
    intl_minutes: float = Field(..., alias="Total intl minutes")
    intl_calls: int = Field(..., alias="Total intl calls")
    intl_charge: float = Field(..., alias="Total intl charge")
    customer_service_calls: int = Field(..., alias="Customer service calls")
    
    class Config:
        populate_by_name = True
        allow_population_by_alias = True

# Define response model
class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: bool
    message: str
    feature_importance: Dict[str, float]

def preprocess_input(customer_data: CustomerData):
    """Preprocess input data for prediction"""
    # Convert to DataFrame with alias names
    data_dict = {}
    for field_name, field_value in customer_data.dict(by_alias=True).items():
        data_dict[field_name] = field_value
        
    df = pd.DataFrame([data_dict])
    
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
        
        # Create message
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
            message=message,
            feature_importance=importance_dict
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/features")
def get_features():
    """Return the list of features used by the model"""
    return {"features": feature_names}