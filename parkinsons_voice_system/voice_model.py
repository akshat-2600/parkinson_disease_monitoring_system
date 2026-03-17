"""
Voice-based Parkinson's Detection Model
Supports Random Forest, XGBoost, and MLP
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score,
    recall_score, mean_absolute_error, mean_squared_error, r2_score
)
from xgboost import XGBClassifier, XGBRegressor
import joblib
import json
from typing import Dict, Tuple, Optional, Union
import os


class VoiceModelBase:
    """Base class for voice-based PD models"""
    
    def __init__(self, model_type: str = 'random_forest', task: str = 'classification'):
        """
        Args:
            model_type: 'random_forest', 'xgboost', or 'mlp'
            task: 'classification' or 'regression'
        """
        self.model_type = model_type
        self.task = task
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.best_params = None
        
    def _get_model(self, params: Optional[Dict] = None):
        """Initialize model safely"""

        # Always get fresh default params
        default_params = self._get_default_params()

        # Override only if params provided
        if params:
            default_params.update(params)

        if self.task == 'classification':
            if self.model_type == 'random_forest':
                return RandomForestClassifier(**default_params, random_state=42)

            elif self.model_type == 'xgboost':
                return XGBClassifier(**default_params, random_state=42,
                                    use_label_encoder=False, eval_metric='logloss')

            elif self.model_type == 'mlp':
                # 🚨 FILTER WRONG PARAMS
                allowed_keys = [
                    'hidden_layer_sizes', 'activation',
                    'solver', 'alpha', 'learning_rate'
                ]
                filtered_params = {k: v for k, v in default_params.items() if k in allowed_keys}

                return MLPClassifier(**filtered_params, random_state=42, max_iter=500)

        else:
            if self.model_type == 'random_forest':
                return RandomForestRegressor(**default_params, random_state=42)

            elif self.model_type == 'xgboost':
                return XGBRegressor(**default_params, random_state=42)

            elif self.model_type == 'mlp':
                allowed_keys = [
                    'hidden_layer_sizes', 'activation',
                    'solver', 'alpha', 'learning_rate'
                ]
                filtered_params = {k: v for k, v in default_params.items() if k in allowed_keys}

                return MLPRegressor(**filtered_params, random_state=42, max_iter=500)
    
    def _get_default_params(self) -> Dict:
        """Get default parameters for each model type"""
        if self.task == 'classification':
            params = {
                'random_forest': {
                    'n_estimators': 200,
                    'max_depth': 20,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'max_features': 'sqrt',
                    'class_weight': 'balanced'
                },
                'xgboost': {
                    'n_estimators': 200,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'scale_pos_weight': 1
                },
                'mlp': {
                    'hidden_layer_sizes': (128, 64, 32),
                    'activation': 'relu',
                    'solver': 'adam',
                    'alpha': 0.0001,
                    'learning_rate': 'adaptive'
                }
            }
        else:  # regression
            params = {
                'random_forest': {
                    'n_estimators': 200,
                    'max_depth': 20,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'max_features': 'sqrt'
                },
                'xgboost': {
                    'n_estimators': 200,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8
                },
                'mlp': {
                    'hidden_layer_sizes': (128, 64, 32),
                    'activation': 'relu',
                    'solver': 'adam',
                    'alpha': 0.0001,
                    'learning_rate': 'adaptive'
                }
            }
            
        return params[self.model_type]
    
    def get_param_grid(self) -> Dict:
        """Get parameter grid for hyperparameter tuning"""
        if self.task == 'classification':
            grids = {
                'random_forest': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2']
                },
                'xgboost': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'subsample': [0.7, 0.8, 0.9],
                    'colsample_bytree': [0.7, 0.8, 0.9]
                },
                'mlp': {
                    'hidden_layer_sizes': [(64,), (128, 64), (128, 64, 32)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive']
                }
            }
        else:
            grids = {
                'random_forest': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'xgboost': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'subsample': [0.7, 0.8, 0.9]
                },
                'mlp': {
                    'hidden_layer_sizes': [(64,), (128, 64), (128, 64, 32)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01]
                }
            }
            
        return grids[self.model_type]
    
    def train(self, X: np.ndarray, y: np.ndarray, 
             tune_hyperparams: bool = False,
             cv_folds: int = 5) -> Dict:
        """
        Train the model
        
        Args:
            X: Feature matrix
            y: Target labels
            tune_hyperparams: Whether to perform hyperparameter tuning
            cv_folds: Number of CV folds for tuning
            
        Returns:
            Training metrics
        """
        # Fit scaler
        X_scaled = self.scaler.fit_transform(X)
        
        if tune_hyperparams:
            print(f"Performing hyperparameter tuning for {self.model_type}...")
            base_model = self._get_model()
            param_grid = self.get_param_grid()
            
            grid_search = GridSearchCV(
                base_model, param_grid,
                cv=cv_folds,
                scoring='roc_auc' if self.task == 'classification' else 'neg_mean_absolute_error',
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_scaled, y)
            
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            print(f"Best parameters: {self.best_params}")
        else:
            self.model = self._get_model()
            self.model.fit(X_scaled, y)
            
        # Calculate training metrics
        y_pred = self.model.predict(X_scaled)
        
        metrics = self._calculate_metrics(y, y_pred, X_scaled)
        
        return metrics
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Evaluate the model
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
            
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        
        metrics = self._calculate_metrics(y, y_pred, X_scaled)
        
        return metrics
    
    def _calculate_metrics(self, y_true: np.ndarray, 
                          y_pred: np.ndarray,
                          X: np.ndarray) -> Dict:
        """Calculate appropriate metrics based on task"""
        if self.task == 'classification':
            # Get probability predictions if available
            if hasattr(self.model, 'predict_proba'):
                y_proba = self.model.predict_proba(X)[:, 1]
                auc = roc_auc_score(y_true, y_proba)
            else:
                auc = None
                
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
                'auc': auc
            }
        else:  # regression
            metrics = {
                'mae': mean_absolute_error(y_true, y_pred),
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'r2': r2_score(y_true, y_pred)
            }
            
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
            
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        Get probability predictions (classification only)
        
        Args:
            X: Feature matrix
            
        Returns:
            Probability predictions or None
        """
        if self.task != 'classification':
            return None
            
        if not hasattr(self.model, 'predict_proba'):
            return None
            
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def save(self, save_dir: str):
        """
        Save model, scaler, and metadata
        
        Args:
            save_dir: Directory to save files
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model
        joblib.dump(self.model, os.path.join(save_dir, 'model.pkl'))
        
        # Save scaler
        joblib.dump(self.scaler, os.path.join(save_dir, 'scaler.pkl'))
        
        # Save metadata
        metadata = {
            'model_type': self.model_type,
            'task': self.task,
            'feature_names': self.feature_names,
            'best_params': self.best_params
        }
        
        with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"Model saved to {save_dir}")
    
    def load(self, save_dir: str):
        """
        Load model, scaler, and metadata
        
        Args:
            save_dir: Directory to load from
        """
        # Load model
        self.model = joblib.load(os.path.join(save_dir, 'model.pkl'))
        
        # Load scaler
        self.scaler = joblib.load(os.path.join(save_dir, 'scaler.pkl'))
        
        # Load metadata
        with open(os.path.join(save_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
            
        self.model_type = metadata['model_type']
        self.task = metadata['task']
        self.feature_names = metadata['feature_names']
        self.best_params = metadata.get('best_params')
        
        print(f"Model loaded from {save_dir}")


class VoiceClassifier(VoiceModelBase):
    """Voice-based PD Classification (Healthy vs PD)"""
    
    def __init__(self, model_type: str = 'xgboost'):
        super().__init__(model_type=model_type, task='classification')


class VoiceSeverityRegressor(VoiceModelBase):
    """Voice-based PD Severity Regression"""
    
    def __init__(self, model_type: str = 'xgboost'):
        super().__init__(model_type=model_type, task='regression')