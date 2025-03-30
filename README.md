# Customer-Churn-Prediction

A complete MLOps implementation for predicting customer churn in a telecommunications company using machine learning.

## Project Overview

This project implements an end-to-end MLOps pipeline for customer churn prediction, from data preprocessing to model deployment and monitoring. The system helps identify customers likely to cancel their subscription, enabling proactive retention strategies.


### Data Preprocessing
- Implemented data cleaning and feature engineering
- Handled missing values and categorical features
- Created preprocessing pipeline for reproducibility
- Tools: Pandas, scikit-learn

### Model Training
- Trained Random Forest classifier for churn prediction
- Implemented hyperparameter tuning
- Created modular training pipeline
- Tools: scikit-learn, joblib

### Evaluation and Visualization
- Implemented model evaluation metrics (accuracy, F1, ROC AUC)
- Created visualizations for model performance
- Generated confusion matrix and feature importance plots
- Tools: Matplotlib, seaborn

### API Development
- Built REST API with FastAPI
- Created endpoints for prediction and model information
- Implemented input validation and error handling
- Tools: FastAPI, Pydantic

### MLflow Integration
- Set up experiment tracking with MLflow
- Logged hyperparameters, metrics, and artifacts
- Implemented model versioning
- Tools: MLflow

### Containerization with Docker
- Containerized the application for consistent deployment
- Created Dockerfiles for both training and serving
- Implemented Docker Compose for local development
- Tools: Docker, Docker Compose

### Monitoring
- Implemented system monitoring with Prometheus
- Created Grafana dashboards for visualization
- Added metrics for model performance and data drift
- Tools: Prometheus, Grafana


## Installation

### Prerequisites
- Python 3.9+
- Docker and Docker Compose
- Git

