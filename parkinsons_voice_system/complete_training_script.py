"""
Complete Training Script
End-to-end training pipeline for voice-based PD detection
"""

import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from voice_trainer import VoiceTrainer, VoiceDataPipeline
from voice_inference import VoiceInference
from voice_explainer import VoiceExplainer


def create_sample_dataset():
    """
    Create sample dataset based on the provided format
    In production, replace with actual data loading
    """
    # Sample data based on the format provided
    data = {
        'name': ['phon_R01_S01_1', 'phon_R01_S01_2'],
        'MDVP:Fo(Hz)': [119.992, 122.4],
        'MDVP:Fhi(Hz)': [157.302, 148.65],
        'MDVP:Flo(Hz)': [74.997, 113.819],
        'MDVP:Jitter(%)': [0.00784, 0.00968],
        'MDVP:Jitter(Abs)': [0.00007, 0.00008],
        'MDVP:RAP': [0.0037, 0.00465],
        'MDVP:PPQ': [0.00554, 0.00696],
        'Jitter:DDP': [0.01109, 0.01394],
        'MDVP:Shimmer': [0.04374, 0.06134],
        'MDVP:Shimmer(dB)': [0.426, 0.626],
        'Shimmer:APQ3': [0.02182, 0.03134],
        'Shimmer:APQ5': [0.0313, 0.04518],
        'MDVP:APQ': [0.02971, 0.04368],
        'Shimmer:DDA': [0.06545, 0.09403],
        'NHR': [0.02211, 0.01929],
        'HNR': [21.033, 19.085],
        'status': [1, 1],  # 1 = PD, 0 = Healthy
        'RPDE': [0.414783, 0.458359],
        'DFA': [0.815285, 0.819521],
        'spread1': [-4.813031, -4.075192],
        'spread2': [0.266482, 0.33559],
        'D2': [2.301442, 2.486855],
        'PPE': [0.284654, 0.368674]
    }
    
    df = pd.DataFrame(data)
    return df


def train_classification_model(data_path: str, 
                               save_dir: str = 'saved_models/voice_model/classification'):
    """
    Train classification model (PD vs Healthy)
    
    Args:
        data_path: Path to dataset CSV
        save_dir: Directory to save model
    """
    print("\n" + "="*80)
    print("TRAINING CLASSIFICATION MODEL")
    print("="*80)
    
    # Initialize trainer
    trainer = VoiceTrainer(task='classification', random_state=42)
    
    # Run training
    results = trainer.run_training(
        data_path=data_path,
        target_col='status',
        n_features=15,  # Select top 15 features
        tune_hyperparams=False,  # Set to True for hyperparameter tuning
        save_dir=save_dir
    )
    
    return results


def train_regression_model(data_path: str,
                           save_dir: str = 'saved_models/voice_model/regression'):
    """
    Train regression model (severity prediction)
    Note: Requires dataset with severity scores (e.g., UPDRS)
    
    Args:
        data_path: Path to dataset CSV with severity scores
        save_dir: Directory to save model
    """
    print("\n" + "="*80)
    print("TRAINING REGRESSION MODEL")
    print("="*80)
    
    # Initialize trainer
    trainer = VoiceTrainer(task='regression', random_state=42)
    
    # Run training
    results = trainer.run_training(
        data_path=data_path,
        target_col='total_UPDRS',  # Or motor_UPDRS, depending on dataset
        n_features=15,
        tune_hyperparams=False,
        save_dir=save_dir
    )
    
    return results


def test_inference(model_dir: str):
    """
    Test inference with the trained model
    
    Args:
        model_dir: Directory containing saved model
    """
    print("\n" + "="*80)
    print("TESTING INFERENCE")
    print("="*80)
    
    # Initialize inference
    inference = VoiceInference(model_dir=model_dir)
    
    # Example features (from the dataset)
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
        'NHR': 0.02211
    }
    
    # Make prediction
    result = inference.predict(example_features, include_explanation=True)
    
    print("\nPrediction Result:")
    print(json.dumps(result, indent=2))
    
    return result


def test_explainability(model_dir: str, data_path: str):
    """
    Test SHAP explainability
    
    Args:
        model_dir: Directory containing saved model
        data_path: Path to dataset for background data
    """
    print("\n" + "="*80)
    print("TESTING EXPLAINABILITY")
    print("="*80)
    
    # Load some background data
    df = pd.read_csv(data_path)
    pipeline = VoiceDataPipeline()
    X, y = pipeline.prepare_features(df, target_col='status')
    X_selected, _ = pipeline.feature_selection(X, y, k=15)
    
    # Initialize explainer with background data
    explainer = VoiceExplainer(
        model_dir=model_dir,
        background_data=X_selected.values[:50]  # Use first 50 samples as background
    )
    
    # Get explanation for one sample
    explanation = explainer.explain_prediction(
        X_selected.values[0],
        patient_id='SAMPLE_001'
    )
    
    print("\nSHAP Explanation:")
    print(json.dumps(explanation, indent=2))
    
    # Create visualizations
    explainer.plot_waterfall(
        X_selected.values[0],
        patient_id='SAMPLE_001',
        save_path='outputs/waterfall_plot.png'
    )
    
    explainer.global_feature_importance(
        X_selected.values[:100],
        save_path='outputs/feature_importance.png'
    )
    
    print("\nExplanation plots saved to outputs/")
    
    return explanation


def generate_training_report(results, save_path: str = 'outputs/training_report.txt'):
    """
    Generate training report
    
    Args:
        results: Training results dictionary
        save_path: Path to save report
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("VOICE-BASED PARKINSON'S DETECTION - TRAINING REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("MODEL COMPARISON\n")
        f.write("-"*80 + "\n\n")
        
        for model_name, metrics in results.items():
            f.write(f"\n{model_name.upper()}\n")
            f.write("-"*40 + "\n")
            
            for split in ['train', 'val', 'test']:
                f.write(f"\n{split.capitalize()} Set:\n")
                for metric, value in metrics[split].items():
                    if value is not None:
                        f.write(f"  {metric}: {value:.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"Training report saved to {save_path}")


def main():
    """
    Main training pipeline
    """
    # Create output directories
    os.makedirs('saved_models/voice_model/classification', exist_ok=True)
    os.makedirs('saved_models/voice_model/regression', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    
    # IMPORTANT: Replace with actual dataset path
    # For demonstration, using placeholder
    data_path = 'data/voice_dataset.csv'
    
    # Check if data exists, otherwise create sample
    if not os.path.exists(data_path):
        print("WARNING: Using sample data. Replace with actual dataset!")
        os.makedirs('data', exist_ok=True)
        sample_df = create_sample_dataset()
        sample_df.to_csv(data_path, index=False)
    
    # 1. Train Classification Model
    try:
        clf_results = train_classification_model(
            data_path=data_path,
            save_dir='saved_models/voice_model/classification'
        )
        
        # Generate report
        generate_training_report(
            clf_results,
            save_path='outputs/classification_training_report.txt'
        )
        
        # 2. Test Inference
        test_inference(model_dir='saved_models/voice_model/classification')
        
        # 3. Test Explainability
        test_explainability(
            model_dir='saved_models/voice_model/classification',
            data_path=data_path
        )
        
    except Exception as e:
        print(f"\nError during training: {e}")
        print("\nPlease ensure you have the actual Parkinson's dataset.")
        print("Expected columns: name, MDVP:Fo(Hz), MDVP:Fhi(Hz), ..., status")
        return
    
    # 4. Train Regression Model (if severity data available)
    # Uncomment if you have UPDRS severity scores
    """
    try:
        reg_results = train_regression_model(
            data_path='data/parkinsons_updrs_data.csv',
            save_dir='saved_models/voice_model/regression'
        )
        
        generate_training_report(
            reg_results,
            save_path='outputs/regression_training_report.txt'
        )
    except Exception as e:
        print(f"\nRegression training skipped: {e}")
    """
    
    print("\n" + "="*80)
    print("TRAINING PIPELINE COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - saved_models/voice_model/classification/")
    print("  - outputs/classification_training_report.txt")
    print("  - outputs/waterfall_plot.png")
    print("  - outputs/feature_importance.png")
    print("\nNext steps:")
    print("  1. Review training metrics in outputs/")
    print("  2. Test inference with voice_inference.py")
    print("  3. Generate explanations with voice_explainer.py")
    print("  4. Integrate into production API")


if __name__ == '__main__':
    main()