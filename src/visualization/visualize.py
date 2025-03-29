import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_feature_importance(model, feature_names, top_n=10, save_dir=None):
    """
    Plot feature importances from the model
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        top_n: Number of top features to show
        save_dir: Directory to save the plot
    """
    if hasattr(model, 'feature_importances_'):
        # Get feature importances
        importances = model.feature_importances_
        
        # Create DataFrame for plotting
        import pandas as pd
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        
        # Sort by importance
        feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False).head(top_n)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
        plt.title('Top Feature Importances')
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, 'feature_importance.png'))
            print(f"Feature importance plot saved to {save_dir}")
        
        plt.show()
    else:
        print("Model doesn't have feature_importances_ attribute")

def plot_confusion_matrix(y_true, y_pred, save_dir=None):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_dir: Directory to save the plot
    """
    from sklearn.metrics import confusion_matrix
    import numpy as np
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Churned', 'Churned'],
                yticklabels=['Not Churned', 'Churned'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
        print(f"Confusion matrix plot saved to {save_dir}")
    
    plt.show()