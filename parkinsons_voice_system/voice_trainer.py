"""
Voice Model Training Pipeline
Handles data preprocessing, model training, and evaluation
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List, Optional
import json
import os
from datetime import datetime

from voice_model import VoiceClassifier, VoiceSeverityRegressor


class VoiceDataPipeline:
    """Data preprocessing and feature engineering"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.feature_names = None
        self.selected_features = None
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load voice features dataset
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame with features
        """
        df = pd.read_csv(filepath)
        print(f"Loaded data: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame, 
                             strategy: str = 'mean') -> pd.DataFrame:
        """
        Handle missing values
        
        Args:
            df: Input dataframe
            strategy: 'mean', 'median', or 'drop'
            
        Returns:
            DataFrame with handled missing values
        """
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            print(f"\nMissing values found:")
            print(missing_counts[missing_counts > 0])
            
            if strategy == 'drop':
                df = df.dropna()
            elif strategy == 'mean':
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            elif strategy == 'median':
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
                
            print(f"After handling: {df.isnull().sum().sum()} missing values")
        else:
            print("No missing values found")
            
        return df
    
    def prepare_features(self, df: pd.DataFrame, 
                        target_col: str = 'status',
                        exclude_cols: List[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare feature matrix and target
        
        Args:
            df: Input dataframe
            target_col: Name of target column
            exclude_cols: Columns to exclude from features
            
        Returns:
            X (features), y (target)
        """
        if exclude_cols is None:
            exclude_cols = ['name']
            
        # Separate features and target
        y = df[target_col]
        
        # Get feature columns
        feature_cols = [col for col in df.columns 
                       if col != target_col and col not in exclude_cols]
        
        X = df[feature_cols]
        
        # Store feature names
        self.feature_names = feature_cols
        
        print(f"\nFeatures: {len(feature_cols)} columns")
        print(f"Target distribution:\n{y.value_counts()}")
        
        return X, y
    
    def feature_selection(self, X: pd.DataFrame, y: pd.Series,
                         method: str = 'mutual_info',
                         k: int = 15,
                         task: str = 'classification') -> Tuple[pd.DataFrame, List[str]]:
        """
        Select top k features
        
        Args:
            X: Feature matrix
            y: Target
            method: 'mutual_info', 'f_test', or 'variance'
            k: Number of features to select
            task: 'classification' or 'regression'
            
        Returns:
            Selected features and feature names
        """
        print(f"\nPerforming feature selection ({method})...")
        
        if method == 'mutual_info':
            if task == 'classification':
                selector = SelectKBest(mutual_info_classif, k=k)
            else:
                selector = SelectKBest(f_regression, k=k)
        elif method == 'f_test':
            if task == 'classification':
                selector = SelectKBest(f_classif, k=k)
            else:
                selector = SelectKBest(f_regression, k=k)
        else:
            # Use variance threshold
            from sklearn.feature_selection import VarianceThreshold
            selector = VarianceThreshold(threshold=0.01)
            
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        if hasattr(selector, 'get_support'):
            selected_mask = selector.get_support()
            selected_features = [f for f, s in zip(self.feature_names, selected_mask) if s]
        else:
            selected_features = self.feature_names[:k]
            
        self.selected_features = selected_features
        
        print(f"Selected {len(selected_features)} features:")
        for i, feat in enumerate(selected_features[:10], 1):
            print(f"  {i}. {feat}")
        if len(selected_features) > 10:
            print(f"  ... and {len(selected_features) - 10} more")
            
        return pd.DataFrame(X_selected, columns=selected_features), selected_features
    
    def split_data(self, X: pd.DataFrame, y: pd.Series,
                  test_size: float = 0.2,
                  val_size: float = 0.1) -> Dict:
        """
        Split data into train/val/test sets
        
        Args:
            X: Feature matrix
            y: Target
            test_size: Fraction for test set
            val_size: Fraction for validation set
            
        Returns:
            Dictionary with train/val/test splits
        """
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=self.random_state, stratify=y_temp
        )
        
        print(f"\nData split:")
        print(f"  Train: {X_train.shape[0]} samples")
        print(f"  Val:   {X_val.shape[0]} samples")
        print(f"  Test:  {X_test.shape[0]} samples")
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }


class VoiceTrainer:
    """Main trainer class for voice models"""
    
    def __init__(self, task: str = 'classification', random_state: int = 42):
        """
        Args:
            task: 'classification' or 'regression'
            random_state: Random seed
        """
        self.task = task
        self.random_state = random_state
        self.pipeline = VoiceDataPipeline(random_state=random_state)
        self.models = {}
        self.best_model = None
        self.results = {}
        
        # Set seeds for reproducibility
        np.random.seed(random_state)
        
    def run_training(self, 
                    data_path: str,
                    target_col: str = 'status',
                    n_features: int = 15,
                    tune_hyperparams: bool = False,
                    save_dir: str = 'saved_models/voice_model') -> Dict:
        """
        Run complete training pipeline
        
        Args:
            data_path: Path to dataset CSV
            target_col: Name of target column
            n_features: Number of features to select
            tune_hyperparams: Whether to tune hyperparameters
            save_dir: Directory to save best model
            
        Returns:
            Training results
        """
        print("="*80)
        print("VOICE-BASED PARKINSON'S DETECTION TRAINING")
        print(f"Task: {self.task.upper()}")
        print("="*80)
        
        # 1. Load data
        print("\n[1/6] Loading data...")
        df = self.pipeline.load_data(data_path)
        
        # 2. Handle missing values
        print("\n[2/6] Handling missing values...")
        df = self.pipeline.handle_missing_values(df, strategy='mean')
        
        # 3. Prepare features
        print("\n[3/6] Preparing features...")
        X, y = self.pipeline.prepare_features(df, target_col=target_col)
        
        # 4. Feature selection
        print("\n[4/6] Feature selection...")
        X_selected, selected_features = self.pipeline.feature_selection(
            X, y, method='mutual_info', k=n_features, task=self.task
        )
        
        # 5. Split data
        print("\n[5/6] Splitting data...")
        data_splits = self.pipeline.split_data(X_selected, y)
        
        # 6. Train models
        print("\n[6/6] Training models...")
        self._train_all_models(data_splits, tune_hyperparams)
        
        # Compare and select best
        self._compare_models()
        
        # Save best model
        if save_dir:
            print(f"\nSaving best model to {save_dir}...")
            self.best_model.feature_names = selected_features
            self.best_model.save(save_dir)
            
            # Save training results
            results_path = os.path.join(save_dir, 'training_results.json')
            with open(results_path, 'w') as f:
                # Convert numpy types to native Python types
                results_serializable = {}
                for model_name, metrics in self.results.items():
                    results_serializable[model_name] = {}

                    for split_name, split_metrics in metrics.items():
                        results_serializable[model_name][split_name] = {
                            k: float(v) if v is not None else None
                            for k, v in split_metrics.items()
                        }
                json.dump(results_serializable, f, indent=2)
                
        return self.results
    
    def _train_all_models(self, data_splits: Dict, tune_hyperparams: bool = False):
        """Train all model types"""
        model_types = ['random_forest', 'xgboost', 'mlp']
        
        for model_type in model_types:
            print(f"\n{'='*60}")
            print(f"Training {model_type.upper()}")
            print(f"{'='*60}")
            
            # Initialize model
            if self.task == 'classification':
                model = VoiceClassifier(model_type=model_type)
            else:
                model = VoiceSeverityRegressor(model_type=model_type)
            
            # Train
            train_metrics = model.train(
                data_splits['X_train'].values,
                data_splits['y_train'].values,
                tune_hyperparams=tune_hyperparams
            )
            
            # Evaluate on validation set
            val_metrics = model.evaluate(
                data_splits['X_val'].values,
                data_splits['y_val'].values
            )
            
            # Evaluate on test set
            test_metrics = model.evaluate(
                data_splits['X_test'].values,
                data_splits['y_test'].values
            )
            
            # Store model and results
            self.models[model_type] = model
            self.results[model_type] = {
                'train': train_metrics,
                'val': val_metrics,
                'test': test_metrics
            }
            
            # Print results
            print(f"\nResults for {model_type}:")
            print(f"  Train: {self._format_metrics(train_metrics)}")
            print(f"  Val:   {self._format_metrics(val_metrics)}")
            print(f"  Test:  {self._format_metrics(test_metrics)}")
    
    def _compare_models(self):
        """Compare models and select best"""
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        
        if self.task == 'classification':
            metric_key = 'f1_score'
            comparison_metric = 'F1 Score'
        else:
            metric_key = 'mae'
            comparison_metric = 'MAE'
            reverse = True  # For MAE, lower is better
        
        # Create comparison table
        comparison = []
        for model_type, results in self.results.items():
            test_metric = results['test'][metric_key]
            comparison.append({
                'Model': model_type,
                comparison_metric: test_metric
            })
        
        comparison_df = pd.DataFrame(comparison)
        comparison_df = comparison_df.sort_values(
            comparison_metric, 
            ascending=(self.task == 'regression')
        )
        
        print("\nTest Set Performance:")
        print(comparison_df.to_string(index=False))
        
        # Select best model
        best_model_name = comparison_df.iloc[0]['Model']
        self.best_model = self.models[best_model_name]
        
        print(f"\n🏆 Best Model: {best_model_name.upper()}")
        print(f"   {comparison_metric}: {comparison_df.iloc[0][comparison_metric]:.4f}")
        
    def _format_metrics(self, metrics: Dict) -> str:
        """Format metrics for printing"""
        if self.task == 'classification':
            return (f"Acc={metrics['accuracy']:.4f}, "
                   f"F1={metrics['f1_score']:.4f}, "
                   f"AUC={metrics.get('auc', 0):.4f}")
        else:
            return (f"MAE={metrics['mae']:.4f}, "
                   f"RMSE={metrics['rmse']:.4f}, "
                   f"R2={metrics['r2']:.4f}")


def main():
    """Example training run"""
    # Classification task
    print("\n" + "="*80)
    print("CLASSIFICATION: PD vs Healthy")
    print("="*80)
    
    trainer_clf = VoiceTrainer(task='classification', random_state=42)
    
    # Note: Replace with actual dataset path
    results_clf = trainer_clf.run_training(
        data_path='data/parkinsons_data.csv',
        target_col='status',
        n_features=15,
        tune_hyperparams=False,  # Set to True for hyperparameter tuning
        save_dir='saved_models/voice_model/classification'
    )
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)


if __name__ == '__main__':
    main()