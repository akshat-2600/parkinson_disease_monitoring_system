"""
Voice Model Explainability using SHAP
Provides per-patient explanations and global feature importance
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import json
import os
from typing import Union

from voice_model import VoiceClassifier, VoiceSeverityRegressor


class VoiceExplainer:
    """
    SHAP-based explainability for voice models
    """
    
    def __init__(self, model_dir: str, background_data: Optional[np.ndarray] = None):
        """
        Initialize explainer
        
        Args:
            model_dir: Directory containing saved model
            background_data: Background dataset for SHAP (optional)
        """
        self.model_dir = model_dir
        self.model = None
        self.metadata = None
        self.explainer = None
        self.background_data = background_data
        
        self.load_model()
        self._initialize_explainer()
    
    def load_model(self):
        """Load model and metadata"""
        # Load metadata
        metadata_path = os.path.join(self.model_dir, 'metadata.json')
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Initialize and load model
        if self.metadata['task'] == 'classification':
            self.model = VoiceClassifier(model_type=self.metadata['model_type'])
        else:
            self.model = VoiceSeverityRegressor(model_type=self.metadata['model_type'])
        
        self.model.load(self.model_dir)
    
    def _initialize_explainer(self):
        """Initialize SHAP explainer based on model type"""
        if self.metadata['model_type'] in ['random_forest', 'xgboost']:
            # Tree-based models: use TreeExplainer
            self.explainer = shap.TreeExplainer(self.model.model)
            self.explainer_type = 'tree'
        else:
            # Other models: use KernelExplainer
            if self.background_data is None:
                print("Warning: KernelExplainer requires background data. "
                      "Explanations may be slow without it.")
            
            # Create prediction function
            def predict_fn(X):
                X_scaled = self.model.scaler.transform(X)
                if self.metadata['task'] == 'classification':
                    if hasattr(self.model.model, 'predict_proba'):
                        return self.model.model.predict_proba(X_scaled)
                    else:
                        return self.model.model.predict(X_scaled)
                else:
                    return self.model.model.predict(X_scaled)
            
            self.explainer = shap.KernelExplainer(
                predict_fn,
                self.background_data if self.background_data is not None else np.zeros((1, len(self.metadata['feature_names'])))
            )
            self.explainer_type = 'kernel'
    
    def explain_prediction(self, X: np.ndarray, 
                          patient_id: Optional[str] = None) -> Dict:
        """
        Get SHAP explanation for a single prediction
        
        Args:
            X: Feature vector (1D or 2D numpy array)
            patient_id: Optional patient identifier
            
        Returns:
            Explanation dictionary with SHAP values
        """
        # Ensure 2D array
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        # Scale features
        X_scaled = self.model.scaler.transform(X)
        
        # Get SHAP values
        if self.explainer_type == 'tree':
            shap_values = self.explainer.shap_values(X_scaled)
        else:
            shap_values = self.explainer.shap_values(X_scaled)
        
        # For classification with binary output, take positive class
        if self.metadata['task'] == 'classification' and isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Get base value (expected value)
        if hasattr(self.explainer, 'expected_value'):
            if isinstance(self.explainer.expected_value, list):
                base_value = self.explainer.expected_value[1]
            else:
                base_value = self.explainer.expected_value
        else:
            base_value = 0.0
        
        # Get top contributing features
        top_features = self._get_top_features(
            shap_values[0] if len(shap_values.shape) > 1 else shap_values,
            X[0],
            top_k=10
        )
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        if self.metadata['task'] == 'classification':
            proba = self.model.predict_proba(X)
            confidence = proba[0][int(prediction)] if proba is not None else None
        else:
            confidence = None
        
        explanation = {
            'patient_id': patient_id,
            'prediction': float(prediction),
            'confidence': float(confidence) if confidence is not None else None,
            'base_value': float(base_value),
            'shap_explanation': {
                'method': 'SHAP',
                'explainer_type': self.explainer_type,
                'top_features': top_features,
                'feature_contributions': self._format_feature_contributions(
                    shap_values[0] if len(shap_values.shape) > 1 else shap_values,
                    X[0]
                )
            },
            'interpretation': self._interpret_explanation(top_features, prediction)
        }
        
        return explanation
    
    def _get_top_features(self, shap_values: np.ndarray, 
                         feature_values: np.ndarray,
                         top_k: int = 10) -> List[Dict]:
        """Get top contributing features"""
        # Get absolute SHAP values for ranking
        abs_shap = np.abs(shap_values)
        top_indices = np.argsort(abs_shap)[-top_k:][::-1]
        
        top_features = []
        for idx in top_indices:
            feature_name = self.metadata['feature_names'][idx]
            shap_val = float(shap_values[idx])
            feature_val = float(feature_values[idx])
            
            top_features.append({
                'feature': feature_name,
                'value': feature_val,
                'shap_value': shap_val,
                'impact': 'increases' if shap_val > 0 else 'decreases',
                'magnitude': 'strong' if abs(shap_val) > 0.1 else 'moderate' if abs(shap_val) > 0.05 else 'weak'
            })
        
        return top_features
    
    def _format_feature_contributions(self, shap_values: np.ndarray,
                                     feature_values: np.ndarray) -> List[Dict]:
        """Format all feature contributions"""
        contributions = []
        
        for idx, (shap_val, feat_val) in enumerate(zip(shap_values, feature_values)):
            contributions.append({
                'feature': self.metadata['feature_names'][idx],
                'value': float(feat_val),
                'shap_value': float(shap_val)
            })
        
        # Sort by absolute SHAP value
        contributions.sort(key=lambda x: abs(x['shap_value']), reverse=True)
        
        return contributions
    
    def _interpret_explanation(self, top_features: List[Dict], 
                              prediction: Union[int, float]) -> str:
        """Generate human-readable interpretation"""
        if self.metadata['task'] == 'classification':
            diagnosis = "Parkinson's Disease" if prediction == 1 else "Healthy"
            
            # Get most influential features
            pos_features = [f for f in top_features[:3] if f['shap_value'] > 0]
            neg_features = [f for f in top_features[:3] if f['shap_value'] < 0]
            
            interpretation = f"The model predicts: {diagnosis}. "
            
            if prediction == 1:
                interpretation += "Key indicators of Parkinson's Disease: "
                if pos_features:
                    features_text = ", ".join([f['feature'] for f in pos_features[:2]])
                    interpretation += f"{features_text}. "
            else:
                interpretation += "Key indicators of healthy status: "
                if neg_features:
                    features_text = ", ".join([f['feature'] for f in neg_features[:2]])
                    interpretation += f"low values in {features_text}. "
        else:
            # Regression interpretation
            severity_category = self._categorize_severity(prediction)
            interpretation = (
                f"Predicted severity score: {prediction:.2f} ({severity_category}). "
                f"Main contributing factors: "
            )
            
            top_3 = top_features[:3]
            for i, feat in enumerate(top_3):
                if i > 0:
                    interpretation += ", "
                interpretation += f"{feat['feature']}"
        
        return interpretation
    
    def _categorize_severity(self, score: float) -> str:
        """Categorize severity score"""
        if score < 20:
            return "minimal"
        elif score < 40:
            return "mild"
        elif score < 60:
            return "moderate"
        else:
            return "severe"
    
    def plot_waterfall(self, X: np.ndarray, 
                      patient_id: Optional[str] = None,
                      save_path: Optional[str] = None,
                      max_display: int = 15):
        """
        Create SHAP waterfall plot for a single prediction
        
        Args:
            X: Feature vector
            patient_id: Patient ID for plot title
            save_path: Path to save plot
            max_display: Maximum features to display
        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        X_scaled = self.model.scaler.transform(X)
        
        # Get SHAP values
        if self.explainer_type == 'tree':
            shap_values = self.explainer.shap_values(X_scaled)
        else:
            shap_values = self.explainer.shap_values(X_scaled)
        
        # For binary classification, take positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Create explanation object
        if hasattr(self.explainer, 'expected_value'):
            if isinstance(self.explainer.expected_value, list):
                expected_value = self.explainer.expected_value[1]
            else:
                expected_value = self.explainer.expected_value
        else:
            expected_value = 0.0
        
        explanation = shap.Explanation(
            values=shap_values[0],
            base_values=expected_value,
            data=X[0],
            feature_names=self.metadata['feature_names']
        )
        
        # Create plot
        plt.figure(figsize=(10, 8))
        shap.plots.waterfall(explanation, max_display=max_display, show=False)
        
        if patient_id:
            plt.title(f'SHAP Waterfall Plot - Patient: {patient_id}', fontsize=14, pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Waterfall plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_force(self, X: np.ndarray,
                  patient_id: Optional[str] = None,
                  save_path: Optional[str] = None):
        """
        Create SHAP force plot
        
        Args:
            X: Feature vector
            patient_id: Patient ID
            save_path: Path to save plot
        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        X_scaled = self.model.scaler.transform(X)
        
        # Get SHAP values
        if self.explainer_type == 'tree':
            shap_values = self.explainer.shap_values(X_scaled)
        else:
            shap_values = self.explainer.shap_values(X_scaled)
        
        # For binary classification, take positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Get expected value
        if hasattr(self.explainer, 'expected_value'):
            if isinstance(self.explainer.expected_value, list):
                expected_value = self.explainer.expected_value[1]
            else:
                expected_value = self.explainer.expected_value
        else:
            expected_value = 0.0
        
        # Create force plot
        shap.force_plot(
            expected_value,
            shap_values[0],
            X[0],
            feature_names=self.metadata['feature_names'],
            matplotlib=True,
            show=False
        )
        
        if patient_id:
            plt.title(f'SHAP Force Plot - Patient: {patient_id}', fontsize=12)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Force plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def global_feature_importance(self, X_background: np.ndarray,
                                 top_k: int = 15,
                                 save_path: Optional[str] = None):
        """
        Calculate and plot global feature importance
        
        Args:
            X_background: Background dataset
            top_k: Number of top features to show
            save_path: Path to save plot
        """
        X_scaled = self.model.scaler.transform(X_background)
        
        # Get SHAP values
        if self.explainer_type == 'tree':
            shap_values = self.explainer.shap_values(X_scaled)
        else:
            shap_values = self.explainer.shap_values(X_scaled)
        
        # For binary classification, take positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Get top features
        top_indices = np.argsort(mean_abs_shap)[-top_k:]
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        feature_names = [self.metadata['feature_names'][i] for i in top_indices]
        importance_values = mean_abs_shap[top_indices]
        
        plt.barh(range(top_k), importance_values, color='skyblue', edgecolor='navy')
        plt.yticks(range(top_k), feature_names)
        plt.xlabel('Mean |SHAP value|', fontsize=12)
        plt.title('Global Feature Importance (SHAP)', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        # Return importance dict
        importance_dict = {
            self.metadata['feature_names'][i]: float(mean_abs_shap[i])
            for i in top_indices[::-1]
        }
        
        return importance_dict


def main():
    """Example usage"""
    # Example: Load model and create explanations
    explainer = VoiceExplainer(
        model_dir='saved_models/voice_model/classification'
    )
    
    # Example features
    example_features = np.array([[
        119.992, 157.302, 74.997, 0.00784, 0.00007, 0.0037, 0.00554,
        0.01109, 0.04374, 0.426, 0.02182, 0.0313, 0.02971, 0.06545,
        0.02211
    ]])  # First 15 features as example
    
    # Get explanation
    explanation = explainer.explain_prediction(example_features, patient_id='P001')
    
    print("\n" + "="*80)
    print("SHAP EXPLANATION")
    print("="*80)
    print(json.dumps(explanation, indent=2))
    
    # Create waterfall plot
    explainer.plot_waterfall(
        example_features,
        patient_id='P001',
        save_path='explanation_waterfall.png'
    )


if __name__ == '__main__':
    main()