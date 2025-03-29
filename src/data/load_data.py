import pandas as pd
import os

def load_dataset(filepath):
    """
    Load dataset from a CSV file
    
    Args:
        filepath: Path to the dataset
    
    Returns:
        pandas.DataFrame: The loaded dataset
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} not found")
    
    df = pd.read_csv(filepath)
    print(f"Dataset loaded with shape: {df.shape}")
    return df