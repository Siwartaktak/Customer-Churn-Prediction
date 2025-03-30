import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from prometheus_client import Gauge

class DriftDetector:
    def __init__(self, reference_data_path):
        """Initialize with reference data"""
        self.reference_data = pd.read_csv(reference_data_path)
        self.drift_scores = Gauge(
            'data_drift_scores',
            'Drift scores for numerical features',
            ['feature_name']
        )
        self.categorical_drift = Gauge(
            'categorical_drift_scores',
            'Distribution change for categorical features',
            ['feature_name', 'category']
        )
        
    def compute_numerical_drift(self, current_data, feature):
        """Compute Kolmogorov-Smirnov statistic for drift detection"""
        if feature not in self.reference_data.columns or feature not in current_data.columns:
            return 0
        
        ref_values = self.reference_data[feature].dropna()
        curr_values = current_data[feature].dropna()
        
        if len(ref_values) < 2 or len(curr_values) < 2:
            return 0
        
        # KS test returns statistic and p-value, we use statistic as drift measure
        statistic, _ = ks_2samp(ref_values, curr_values)
        self.drift_scores.labels(feature_name=feature).set(statistic)
        return statistic
        
    def compute_categorical_drift(self, current_data, feature):
        """Compute distribution change for categorical features"""
        if feature not in self.reference_data.columns or feature not in current_data.columns:
            return {}
            
        ref_dist = self.reference_data[feature].value_counts(normalize=True).to_dict()
        curr_dist = current_data[feature].value_counts(normalize=True).to_dict()
        
        drift_by_category = {}
        all_categories = set(ref_dist.keys()) | set(curr_dist.keys())
        
        for category in all_categories:
            ref_freq = ref_dist.get(category, 0)
            curr_freq = curr_dist.get(category, 0)
            drift = abs(ref_freq - curr_freq)
            drift_by_category[category] = drift
            
            # Record to Prometheus
            self.categorical_drift.labels(
                feature_name=feature, 
                category=str(category)
            ).set(drift)
            
        return drift_by_category