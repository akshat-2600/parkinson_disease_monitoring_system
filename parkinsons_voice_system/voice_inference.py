"""
Voice Model Inference
Production-ready inference with standardized JSON output
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
import json
import os
from datetime import datetime

from voice_model import VoiceClassifier, VoiceSeverityRegressor


class VoiceInference:
    """
    Production-ready inference class for voice-based PD detection
    Returns standardized JSON format
    """
    
    def __init__(self, model_dir: str):
        """
        Initialize inference engine
        
        Args:
            model_dir: Directory containing saved model
        """
        self.model_dir = model_dir
        self.model = None
        self.metadata = None
        self.load_model()
        
    def load_model(self):
        """Load model and metadata"""
        # Load metadata to determine task type
        metadata_path = os.path.join(self.model_dir, 'metadata.json')
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Initialize appropriate model class
        if self.metadata['task'] == 'classification':
            self.model = VoiceClassifier(model_type=self.metadata['model_type'])
        else:
            self.model = VoiceSeverityRegressor(model_type=self.metadata['model_type'])
        
        # Load trained model
        self.model.load(self.model_dir)
        
        print(f"Loaded {self.metadata['model_type']} model for {self.metadata['task']}")
        print(f"Expected features: {len(self.metadata['feature_names'])}")
    
    def predict(self, features: Union[Dict, pd.DataFrame, np.ndarray],
                include_explanation: bool = False) -> Dict:
        """
        Make prediction with standardized JSON output
        
        Args:
            features: Input features (dict, DataFrame, or numpy array)
            include_explanation: Whether to include explainability info
            
        Returns:
            Standardized JSON response:
            {
                "prediction": value,
                "confidence": value,
                "explanation": {...},
                "metadata": {...}
            }
        """
        # Convert input to numpy array
        X = self._prepare_input(features)
        
        # Make prediction
        if self.metadata['task'] == 'classification':
            prediction = int(self.model.predict(X)[0])
            proba = self.model.predict_proba(X)
            
            if proba is not None:
                confidence = float(proba[0][prediction])
                class_probabilities = {
                    'healthy': float(proba[0][0]),
                    'parkinsons': float(proba[0][1])
                }
            else:
                confidence = None
                class_probabilities = None
            
            # Create response
            response = {
                "prediction": prediction,
                "prediction_label": "Parkinson's Disease" if prediction == 1 else "Healthy",
                "confidence": confidence,
                "class_probabilities": class_probabilities,
                "explanation": self._get_explanation(X, prediction) if include_explanation else None,
                "metadata": {
                    "model_type": self.metadata['model_type'],
                    "task": self.metadata['task'],
                    "timestamp": datetime.now().isoformat(),
                    "version": "1.0.0"
                }
            }
            
        else:  # regression
            prediction = float(self.model.predict(X)[0])
            
            # For regression, confidence can be based on prediction interval
            # or model variance (simplified here)
            confidence = self._estimate_regression_confidence(prediction)
            
            response = {
                "prediction": prediction,
                "prediction_label": self._get_severity_label(prediction),
                "confidence": confidence,
                "severity_category": self._categorize_severity(prediction),
                "explanation": self._get_explanation(X, prediction) if include_explanation else None,
                "metadata": {
                    "model_type": self.metadata['model_type'],
                    "task": self.metadata['task'],
                    "timestamp": datetime.now().isoformat(),
                    "version": "1.0.0"
                }
            }
        
        return response
    
    def predict_batch(self, features_list: List[Union[Dict, pd.DataFrame]],
                     include_explanation: bool = False) -> List[Dict]:
        """
        Batch prediction
        
        Args:
            features_list: List of feature inputs
            include_explanation: Whether to include explanations
            
        Returns:
            List of prediction responses
        """
        results = []
        for features in features_list:
            result = self.predict(features, include_explanation=include_explanation)
            results.append(result)
        
        return results
    
    def _prepare_input(self, features: Union[Dict, pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Convert input to numpy array in correct format"""
        if isinstance(features, dict):
            # Convert dict to DataFrame then to numpy
            # Ensure correct feature order
            feature_values = [features.get(fname, 0.0) for fname in self.metadata['feature_names']]
            X = np.array([feature_values])
            
        elif isinstance(features, pd.DataFrame):
            # Ensure correct column order
            X = features[self.metadata['feature_names']].values
            
        elif isinstance(features, np.ndarray):
            # Assume already in correct format
            if len(features.shape) == 1:
                X = features.reshape(1, -1)
            else:
                X = features
        else:
            raise ValueError(f"Unsupported input type: {type(features)}")
        
        # Validate feature count
        if X.shape[1] != len(self.metadata['feature_names']):
            raise ValueError(
                f"Expected {len(self.metadata['feature_names'])} features, "
                f"got {X.shape[1]}"
            )
        
        return X
    
    def _get_explanation(self, X: np.ndarray, prediction: Union[int, float]) -> Dict:
        """
        Get basic explanation (feature importance)
        For detailed SHAP explanations, use voice_explainer.py
        """
        # Get feature importances if available
        if hasattr(self.model.model, 'feature_importances_'):
            importances = self.model.model.feature_importances_
            
            # Get top features
            top_indices = np.argsort(importances)[-5:][::-1]
            top_features = []
            
            for idx in top_indices:
                top_features.append({
                    'feature': self.metadata['feature_names'][idx],
                    'importance': float(importances[idx]),
                    'value': float(X[0][idx])
                })
            
            return {
                'method': 'feature_importance',
                'top_features': top_features,
                'note': 'Use voice_explainer.py for detailed SHAP explanations'
            }
        else:
            return {
                'method': 'not_available',
                'note': 'Feature importances not available for this model type'
            }
    
    def _estimate_regression_confidence(self, prediction: float) -> float:
        """
        Estimate confidence for regression prediction
        Simplified approach - can be enhanced with prediction intervals
        """
        # Normalize prediction to 0-1 range (assuming UPDRS-like scale)
        # This is a simplified approach
        normalized = min(max(prediction / 100.0, 0), 1)
        
        # Higher confidence for mid-range predictions
        # Lower for extreme values
        if 0.2 <= normalized <= 0.8:
            confidence = 0.85
        elif 0.1 <= normalized <= 0.9:
            confidence = 0.75
        else:
            confidence = 0.65
        
        return confidence
    
    def _get_severity_label(self, severity_score: float) -> str:
        """Convert severity score to descriptive label"""
        if severity_score < 20:
            return "Minimal/No PD symptoms"
        elif severity_score < 40:
            return "Mild PD symptoms"
        elif severity_score < 60:
            return "Moderate PD symptoms"
        else:
            return "Severe PD symptoms"
    
    def _categorize_severity(self, severity_score: float) -> str:
        """Categorize severity into standard categories"""
        if severity_score < 20:
            return "minimal"
        elif severity_score < 40:
            return "mild"
        elif severity_score < 60:
            return "moderate"
        else:
            return "severe"
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            "model_type": self.metadata['model_type'],
            "task": self.metadata['task'],
            "n_features": len(self.metadata['feature_names']),
            "feature_names": self.metadata['feature_names'],
            "best_params": self.metadata.get('best_params'),
            "version": "1.0.0"
        }


def main():
    """Example usage"""
    # Example voice features (from dataset format)
    example_features = {
        'MDVP:Fo(Hz)': 119.992,
        'MDVP:Fhi(Hz)': 157.302,
        'MDVP:Flo(Hz)': 74.997,
        'MDVP:Jitter(%)': 0.00784,
        'MDVP:Jitter(Abs)': 0.00007,
        'MDVP:RAP': 0.0037,
        'MDVP:PPQ': 0.00554,
        'Jitter:DDP': 0.01109,
        'MDVP:Shimmer': 0.04374,
        'MDVP:Shimmer(dB)': 0.426,
        'Shimmer:APQ3': 0.02182,
        'Shimmer:APQ5': 0.0313,
        'MDVP:APQ': 0.02971,
        'Shimmer:DDA': 0.06545,
        'NHR': 0.02211,
        'HNR': 21.033,
        'RPDE': 0.414783,
        'DFA': 0.815285,
        'spread1': -4.813031,
        'spread2': 0.266482,
        'D2': 2.301442,
        'PPE': 0.284654
    }
    
    # Initialize inference engine
    inference = VoiceInference(model_dir='saved_models/voice_model/classification')
    
    # Make prediction
    result = inference.predict(example_features, include_explanation=True)
    
    # Print formatted output
    print("\n" + "="*80)
    print("PREDICTION RESULT")
    print("="*80)
    print(json.dumps(result, indent=2))
    
    # Get model info
    print("\n" + "="*80)
    print("MODEL INFORMATION")
    print("="*80)
    model_info = inference.get_model_info()
    print(json.dumps(model_info, indent=2))


if __name__ == '__main__':
    main()
    