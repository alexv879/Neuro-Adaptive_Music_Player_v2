"""
Example Script: Automated Hyperparameter Tuning
================================================

This script demonstrates how to automatically optimize your
emotion recognition model hyperparameters.

Prerequisites:
    pip install keras-tuner optuna

Usage:
    # Option 1: Keras Tuner (Recommended - Easy to use)
    python run_hyperparameter_tuning.py --method keras-tuner --max-trials 30
    
    # Option 2: Optuna (Advanced - Most efficient)
    python run_hyperparameter_tuning.py --method optuna --max-trials 50
    
    # Option 3: Quick test (5 trials)
    python run_hyperparameter_tuning.py --method keras-tuner --max-trials 5

Results:
    - Best hyperparameters saved to models/best_hyperparameters.json
    - Apply them to src/config.py for production use
    
Time estimates:
    - 5 trials: ~10-15 minutes
    - 30 trials: ~1-2 hours
    - 100 trials: ~3-5 hours
    
Author: CMP9780M Assessment
"""

import numpy as np
import argparse
from pathlib import Path
import sys
import logging

# Setup paths
sys.path.append(str(Path(__file__).parent / "src"))

from hyperparameter_tuner import (
    KerasTunerOptimizer,
    OptunaOptimizer,
    prepare_data_for_tuning
)
from config import MODEL_DIR, DATA_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_dummy_data(n_samples: int = 1000, n_features: int = 167, n_classes: int = 5):
    """
    Generate dummy EEG data for testing.
    
    Replace this with your actual data loading code.
    """
    logger.info(f"Generating dummy data: {n_samples} samples, {n_features} features...")
    
    np.random.seed(42)
    X_train = np.random.randn(n_samples, n_features).astype(np.float32)
    y_train = np.random.randint(0, n_classes, n_samples)
    
    X_val = np.random.randn(n_samples // 4, n_features).astype(np.float32)
    y_val = np.random.randint(0, n_classes, n_samples // 4)
    
    return X_train, y_train, X_val, y_val


def load_real_data():
    """
    Load your actual preprocessed EEG data.
    
    Expected format:
        - X: shape (n_samples, n_features) - extracted features
        - y: shape (n_samples,) - emotion labels (0-4)
    
    Returns:
        X_train, y_train, X_val, y_val
    """
    # TODO: Replace with your actual data loading
    # Example:
    # from preprocessing import load_preprocessed_data
    # X_train, y_train, X_val, y_val = load_preprocessed_data()
    
    logger.warning("⚠️  Using dummy data. Replace load_real_data() with actual data loading!")
    return generate_dummy_data()


def main():
    parser = argparse.ArgumentParser(
        description='Automated Hyperparameter Optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (5 trials, ~10 minutes)
  python %(prog)s --method keras-tuner --max-trials 5
  
  # Full optimization (50 trials, ~2 hours)
  python %(prog)s --method optuna --max-trials 50
  
  # Bayesian optimization (most intelligent)
  python %(prog)s --method keras-tuner --tuning-algorithm bayesian --max-trials 30
        """
    )
    
    parser.add_argument(
        '--method',
        type=str,
        default='keras-tuner',
        choices=['keras-tuner', 'optuna'],
        help='Optimization method (default: keras-tuner)'
    )
    
    parser.add_argument(
        '--max-trials',
        type=int,
        default=30,
        help='Maximum number of trials (default: 30)'
    )
    
    parser.add_argument(
        '--tuning-algorithm',
        type=str,
        default='bayesian',
        choices=['bayesian', 'hyperband', 'random'],
        help='Keras Tuner algorithm (default: bayesian)'
    )
    
    parser.add_argument(
        '--use-dummy-data',
        action='store_true',
        help='Use dummy data for testing'
    )
    
    args = parser.parse_args()
    
    # Header
    print("=" * 80)
    print("🚀 AUTOMATED HYPERPARAMETER OPTIMIZATION")
    print("=" * 80)
    print(f"Method: {args.method}")
    print(f"Max trials: {args.max_trials}")
    if args.method == 'keras-tuner':
        print(f"Algorithm: {args.tuning_algorithm}")
    print("=" * 80)
    print()
    
    # Load data
    logger.info("📊 Loading data...")
    if args.use_dummy_data:
        X_train, y_train, X_val, y_val = generate_dummy_data()
    else:
        X_train, y_train, X_val, y_val = load_real_data()
    
    logger.info(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")
    
    # Prepare hierarchical labels
    logger.info("🔄 Preparing hierarchical labels...")
    _, y_train_dict = prepare_data_for_tuning(X_train, y_train)
    _, y_val_dict = prepare_data_for_tuning(X_val, y_val)
    
    # Run optimization
    if args.method == 'keras-tuner':
        logger.info(f"🔍 Starting Keras Tuner ({args.tuning_algorithm} optimization)...")
        
        optimizer = KerasTunerOptimizer(
            input_shape=(X_train.shape[1],),
            project_name=f"emotion_tuning_{args.tuning_algorithm}"
        )
        
        tuner = optimizer.tune(
            X_train, y_train_dict,
            X_val, y_val_dict,
            method=args.tuning_algorithm,
            max_trials=args.max_trials,
            executions_per_trial=1
        )
        
        # Get results
        best_hps = optimizer.get_best_hyperparameters(tuner)
        
        print("\n" + "=" * 80)
        print("🎉 OPTIMIZATION COMPLETE!")
        print("=" * 80)
        print("\n📊 BEST HYPERPARAMETERS:")
        print("-" * 80)
        for key, value in best_hps.items():
            print(f"  {key:30s}: {value}")
        
        # Get best model
        best_models = tuner.get_best_models(num_models=1)
        best_model = best_models[0]
        
        # Evaluate
        results = best_model.evaluate(X_val, y_val_dict, verbose=0)
        print("\n📈 VALIDATION PERFORMANCE:")
        print("-" * 80)
        print(f"  Emotion Accuracy: {results[4]:.4f} ({results[4]*100:.2f}%)")
        print(f"  Valence Accuracy: {results[2]:.4f}")
        print(f"  Arousal Accuracy: {results[3]:.4f}")
        
        # Save
        optimizer.save_best_config(tuner)
        
    elif args.method == 'optuna':
        logger.info("🚀 Starting Optuna optimization (TPE algorithm)...")
        
        optimizer = OptunaOptimizer(
            input_shape=(X_train.shape[1],),
            study_name="emotion_recognition_optuna"
        )
        
        study = optimizer.optimize(
            X_train, y_train_dict,
            X_val, y_val_dict,
            n_trials=args.max_trials
        )
        
        # Get results
        print("\n" + "=" * 80)
        print("🎉 OPTIMIZATION COMPLETE!")
        print("=" * 80)
        print("\n📊 BEST HYPERPARAMETERS:")
        print("-" * 80)
        for key, value in study.best_params.items():
            print(f"  {key:30s}: {value}")
        
        print("\n📈 VALIDATION PERFORMANCE:")
        print("-" * 80)
        print(f"  Best Accuracy: {study.best_value:.4f} ({study.best_value*100:.2f}%)")
        
        # Save
        optimizer.save_best_config()
    
    print("\n" + "=" * 80)
    print("💾 NEXT STEPS:")
    print("=" * 80)
    print("1. Check models/best_hyperparameters.json for optimal values")
    print("2. Update src/config.py with these hyperparameters")
    print("3. Retrain your model with optimized settings")
    print("4. Expected improvement: 5-15% accuracy boost")
    print("=" * 80)
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Optimization interrupted by user")
        print("Partial results may be available in models/tuning/")
    except Exception as e:
        logger.error(f"❌ Error during optimization: {e}", exc_info=True)
        sys.exit(1)
