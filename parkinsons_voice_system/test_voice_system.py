"""
Unit Tests for Voice-based PD Detection System
"""

import pytest
import numpy as np
import pandas as pd
import os
import tempfile
import json

from voice_model import VoiceClassifier, VoiceSeverityRegressor
from voice_trainer import VoiceDataPipeline, VoiceTrainer
from voice_inference import VoiceInference


class TestVoiceModel:
    """Test voice model classes"""
    
    def setup_method(self):
        """Setup test data"""
        np.random.seed(42)
        self.X_train = np.random.randn(100, 15)
        self.y_train_clf = np.random.randint(0, 2, 100)
        self.y_train_reg = np.random.randn(100) * 20 + 40
        
    def test_classifier_initialization(self):
        """Test classifier initialization"""
        model = VoiceClassifier(model_type='random_forest')
        assert model.task == 'classification'
        assert model.model_type == 'random_forest'
        
    def test_classifier_training(self):
        """Test classifier training"""
        model = VoiceClassifier(model_type='random_forest')
        metrics = model.train(self.X_train, self.y_train_clf)
        
        assert 'accuracy' in metrics
        assert 'f1_score' in metrics
        assert model.model is not None
        
    def test_classifier_prediction(self):
        """Test classifier prediction"""
        model = VoiceClassifier(model_type='random_forest')
        model.train(self.X_train, self.y_train_clf)
        
        predictions = model.predict(self.X_train[:10])
        assert len(predictions) == 10
        assert all(p in [0, 1] for p in predictions)
        
    def test_regressor_initialization(self):
        """Test regressor initialization"""
        model = VoiceSeverityRegressor(model_type='xgboost')
        assert model.task == 'regression'
        assert model.model_type == 'xgboost'
        
    def test_regressor_training(self):
        """Test regressor training"""
        model = VoiceSeverityRegressor(model_type='random_forest')
        metrics = model.train(self.X_train, self.y_train_reg)
        
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'r2' in metrics
        
    def test_model_save_load(self):
        """Test model save and load"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Train and save
            model = VoiceClassifier(model_type='random_forest')
            model.feature_names = [f'feat_{i}' for i in range(15)]
            model.train(self.X_train, self.y_train_clf)
            model.save(tmpdir)
            
            # Check files exist
            assert os.path.exists(os.path.join(tmpdir, 'model.pkl'))
            assert os.path.exists(os.path.join(tmpdir, 'scaler.pkl'))
            assert os.path.exists(os.path.join(tmpdir, 'metadata.json'))
            
            # Load and test
            model2 = VoiceClassifier()
            model2.load(tmpdir)
            
            assert model2.model_type == 'random_forest'
            assert model2.task == 'classification'
            
            # Predictions should match
            pred1 = model.predict(self.X_train[:5])
            pred2 = model2.predict(self.X_train[:5])
            np.testing.assert_array_equal(pred1, pred2)


class TestDataPipeline:
    """Test data pipeline"""
    
    def setup_method(self):
        """Setup test dataframe"""
        self.df = pd.DataFrame({
            'name': ['P001', 'P002', 'P003'],
            'feat1': [1.0, 2.0, 3.0],
            'feat2': [4.0, 5.0, 6.0],
            'feat3': [7.0, np.nan, 9.0],
            'status': [1, 0, 1]
        })
        
    def test_handle_missing_values_mean(self):
        """Test missing value handling with mean strategy"""
        pipeline = VoiceDataPipeline()
        df_clean = pipeline.handle_missing_values(self.df, strategy='mean')
        
        assert df_clean['feat3'].isnull().sum() == 0
        assert df_clean['feat3'].iloc[1] == 8.0  # Mean of 7 and 9
        
    def test_handle_missing_values_drop(self):
        """Test missing value handling with drop strategy"""
        pipeline = VoiceDataPipeline()
        df_clean = pipeline.handle_missing_values(self.df, strategy='drop')
        
        assert len(df_clean) == 2
        assert df_clean['feat3'].isnull().sum() == 0
        
    def test_prepare_features(self):
        """Test feature preparation"""
        pipeline = VoiceDataPipeline()
        X, y = pipeline.prepare_features(self.df, target_col='status')
        
        assert X.shape[1] == 3  # 3 features (feat1, feat2, feat3)
        assert len(y) == 3
        assert 'name' not in X.columns
        
    def test_split_data(self):
        """Test data splitting"""
        # Create larger dataset
        np.random.seed(42)
        df_large = pd.DataFrame({
            'feat1': np.random.randn(100),
            'feat2': np.random.randn(100),
            'status': np.random.randint(0, 2, 100)
        })
        
        pipeline = VoiceDataPipeline()
        X, y = pipeline.prepare_features(df_large, target_col='status')
        splits = pipeline.split_data(X, y, test_size=0.2, val_size=0.1)
        
        assert 'X_train' in splits
        assert 'X_val' in splits
        assert 'X_test' in splits
        
        total_samples = len(splits['X_train']) + len(splits['X_val']) + len(splits['X_test'])
        assert total_samples == 100


class TestInference:
    """Test inference module"""
    
    def setup_method(self):
        """Setup test model"""
        np.random.seed(42)
        self.X_train = np.random.randn(100, 15)
        self.y_train = np.random.randint(0, 2, 100)
        
        # Create and save temporary model
        self.tmpdir = tempfile.mkdtemp()
        model = VoiceClassifier(model_type='random_forest')
        model.feature_names = [f'feat_{i}' for i in range(15)]
        model.train(self.X_train, self.y_train)
        model.save(self.tmpdir)
        
    def teardown_method(self):
        """Cleanup"""
        import shutil
        shutil.rmtree(self.tmpdir)
        
    def test_inference_initialization(self):
        """Test inference initialization"""
        inference = VoiceInference(self.tmpdir)
        
        assert inference.model is not None
        assert inference.metadata is not None
        
    def test_inference_dict_input(self):
        """Test prediction with dict input"""
        inference = VoiceInference(self.tmpdir)
        
        features = {f'feat_{i}': float(i) for i in range(15)}
        result = inference.predict(features)
        
        assert 'prediction' in result
        assert 'confidence' in result
        assert 'metadata' in result
        
    def test_inference_array_input(self):
        """Test prediction with array input"""
        inference = VoiceInference(self.tmpdir)
        
        features = np.random.randn(1, 15)
        result = inference.predict(features)
        
        assert 'prediction' in result
        assert result['prediction'] in [0, 1]
        
    def test_inference_output_format(self):
        """Test standardized output format"""
        inference = VoiceInference(self.tmpdir)
        
        features = {f'feat_{i}': float(i) for i in range(15)}
        result = inference.predict(features, include_explanation=True)
        
        # Check required fields
        assert 'prediction' in result
        assert 'prediction_label' in result
        assert 'confidence' in result
        assert 'explanation' in result
        assert 'metadata' in result
        
        # Check metadata fields
        assert 'model_type' in result['metadata']
        assert 'task' in result['metadata']
        assert 'timestamp' in result['metadata']
        assert 'version' in result['metadata']
        
    def test_batch_prediction(self):
        """Test batch prediction"""
        inference = VoiceInference(self.tmpdir)
        
        features_list = [
            {f'feat_{i}': float(i) for i in range(15)},
            {f'feat_{i}': float(i+1) for i in range(15)},
        ]
        
        results = inference.predict_batch(features_list)
        
        assert len(results) == 2
        assert all('prediction' in r for r in results)


def test_end_to_end_pipeline():
    """Test complete end-to-end pipeline"""
    # Create sample data
    np.random.seed(42)
    df = pd.DataFrame({
        'name': [f'P{i:03d}' for i in range(100)],
        **{f'feat_{i}': np.random.randn(100) for i in range(20)},
        'status': np.random.randint(0, 2, 100)
    })
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save data
        data_path = os.path.join(tmpdir, 'data.csv')
        df.to_csv(data_path, index=False)
        
        # Train
        trainer = VoiceTrainer(task='classification', random_state=42)
        results = trainer.run_training(
            data_path=data_path,
            target_col='status',
            n_features=10,
            tune_hyperparams=False,
            save_dir=os.path.join(tmpdir, 'model')
        )
        
        # Check results
        assert 'random_forest' in results
        assert 'xgboost' in results
        assert 'mlp' in results
        
        # Test inference
        inference = VoiceInference(os.path.join(tmpdir, 'model'))
        features = {f'feat_{i}': float(i) for i in range(10)}
        result = inference.predict(features)
        
        assert result['prediction'] in [0, 1]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])