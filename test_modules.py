# test_modules.py
import os
import sys
# Add src to path to make imports work
sys.path.append(os.path.abspath('.'))

# Import functions from modules
from src.data.load_data import load_dataset
from src.data.preprocess import preprocess_data
from src.models.train import train_model, evaluate_model
from src.visualization.visualize import plot_feature_importance, plot_confusion_matrix
from sklearn.model_selection import train_test_split

def test_workflow():
    print("Testing the modularized workflow...")
    
    # Step 1: Load the data
    print("\nStep 1: Loading data...")
    train_data_path = "data/raw/train.csv"
    df = load_dataset(train_data_path)
    
    # Step 2: Preprocess the data
    print("\nStep 2: Preprocessing data...")
    X, y, feature_names, encoders = preprocess_data(df)
    
    # Split into train and test sets for testing
    print("\nSplitting data for testing...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Step 3: Train the model
    print("\nStep 3: Training model...")
    model = train_model(X_train, y_train, log_mlflow=False)
    
    # Step 4: Evaluate the model
    print("\nStep 4: Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test, log_mlflow=False)
    
    # Step 5: Visualize results
    print("\nStep 5: Creating visualizations...")
    y_pred = model.predict(X_test)
    plot_feature_importance(model, feature_names, save_dir="output")
    plot_confusion_matrix(y_test, y_pred, save_dir="output")
    
    print("\nTesting complete! Check the output directory for visualizations.")
    
if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    test_workflow()