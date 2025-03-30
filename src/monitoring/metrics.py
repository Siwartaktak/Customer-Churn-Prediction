from prometheus_client import Counter, Histogram, Gauge, Summary
import time

class ModelMonitor:
    def __init__(self):
        # Request metrics
        self.prediction_requests = Counter(
            'churn_prediction_requests_total', 
            'Total number of prediction requests'
        )
        
        # Latency metrics
        self.prediction_latency = Histogram(
            'churn_prediction_latency_seconds', 
            'Time spent processing prediction request',
            buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0)
        )
        
        # Prediction distribution
        self.churn_predictions = Counter(
            'churn_predictions_total', 
            'Distribution of churn predictions',
            ['prediction'] 
        )
        
        # Confidence scores
        self.confidence_scores = Histogram(
            'churn_confidence_scores', 
            'Distribution of confidence scores',
            buckets=(0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0)
        )
        
        # Feature value tracking
        self.feature_values = Gauge(
            'feature_values',
            'Current average values of important features',
            ['feature_name']
        )
        
    def track_prediction(self, start_time, prediction, probability):
        """Track a single prediction"""
        # Record request
        self.prediction_requests.inc()
        
        # Record latency
        elapsed = time.time() - start_time
        self.prediction_latency.observe(elapsed)
        
        # Record prediction
        self.churn_predictions.labels(prediction=str(prediction)).inc()
        
        # Record confidence score
        confidence = max(probability, 1-probability)
        self.confidence_scores.observe(confidence)
        
    def update_feature_tracking(self, feature_name, value):
        """Update the feature value tracker"""
        self.feature_values.labels(feature_name=feature_name).set(value)