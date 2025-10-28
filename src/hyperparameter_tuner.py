"""
Hyperparameter Optimization for Emotion Recognition Model
==========================================================

Automated hyperparameter tuning using:
1. Keras Tuner (Bayesian Optimization, Hyperband)
2. Optuna (Tree-structured Parzen Estimator)
3. Grid/Random Search (Baseline)

This script automatically finds optimal:
- Learning rate, batch size, epochs
- CNN filters, kernel sizes, dropout rates
- LSTM units, recurrent dropout
- Dense layer sizes and dropout
- Callback configurations

Usage:
    python hyperparameter_tuner.py --method keras-tuner --max-trials 50
    python hyperparameter_tuner.py --method optuna --n-trials 100
    python hyperparameter_tuner.py --method grid-search

Author: CMP9780M Assessment
License: Proprietary
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import json
import argparse
from datetime import datetime

# Try importing optimization libraries
KERAS_TUNER_AVAILABLE = False
OPTUNA_AVAILABLE = False

try:
    import keras_tuner as kt
    KERAS_TUNER_AVAILABLE = True
except ImportError:
    print("⚠️  Keras Tuner not installed. Install with: pip install keras-tuner")

try:
    import optuna
    from optuna.integration import TFKerasPruningCallback
    OPTUNA_AVAILABLE = True
except ImportError:
    print("⚠️  Optuna not installed. Install with: pip install optuna")

# Standard imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks, optimizers, regularizers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("❌ TensorFlow required for hyperparameter tuning")

# Local imports
import sys
sys.path.append(str(Path(__file__).parent))
from config import MODEL_DIR, LOG_DIR, EMOTION_CLASSES

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# 1. KERAS TUNER IMPLEMENTATION (RECOMMENDED)
# =============================================================================

class KerasTunerOptimizer:
    """
    Hyperparameter optimization using Keras Tuner.
    
    Supports:
    - Bayesian Optimization (intelligent search)
    - Hyperband (resource-efficient)
    - Random Search (baseline)
    
    Advantages:
    - Native Keras integration
    - Built-in early stopping
    - Automatic resource management
    - Easy to use
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        n_classes: int = EMOTION_CLASSES,
        project_name: str = "emotion_recognition_tuning"
    ):
        if not KERAS_TUNER_AVAILABLE:
            raise ImportError("Keras Tuner not installed")
        
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.project_name = project_name
        self.tuner_dir = MODEL_DIR / "tuning" / project_name
        self.tuner_dir.mkdir(parents=True, exist_ok=True)
        
    def build_model(self, hp: kt.HyperParameters) -> keras.Model:
        """
        Build model with tunable hyperparameters.
        
        Args:
            hp: Keras Tuner HyperParameters object
            
        Returns:
            Compiled Keras model
        """
        # Input layer
        inputs = layers.Input(shape=self.input_shape, name='feature_input')
        
        # Reshape for CNN if needed
        if len(self.input_shape) == 1:
            x = layers.Reshape((self.input_shape[0], 1))(inputs)
        else:
            x = inputs
        
        # ===== CNN HYPERPARAMETERS =====
        
        # Number of CNN blocks
        n_cnn_blocks = hp.Int('n_cnn_blocks', min_value=2, max_value=4, default=3)
        
        for i in range(n_cnn_blocks):
            # Filter size (increasing pattern)
            filters = hp.Int(
                f'cnn_filters_{i}',
                min_value=32,
                max_value=256,
                step=32,
                default=64 * (2 ** i)
            )
            
            # Kernel size
            kernel_size = hp.Choice(f'kernel_size_{i}', values=[3, 5, 7], default=5)
            
            # Conv1D layer
            x = layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                padding='same',
                activation='relu',
                kernel_regularizer=regularizers.l2(hp.Float(
                    f'l2_reg_{i}',
                    min_value=1e-4,
                    max_value=1e-2,
                    sampling='log',
                    default=1e-3
                )),
                name=f'conv1d_{i}'
            )(x)
            
            # Batch normalization
            x = layers.BatchNormalization(name=f'bn_{i}')(x)
            
            # Pooling
            pool_size = hp.Choice(f'pool_size_{i}', values=[2, 3], default=2)
            x = layers.MaxPooling1D(pool_size=pool_size, name=f'maxpool_{i}')(x)
            
            # Dropout
            dropout_rate = hp.Float(
                f'cnn_dropout_{i}',
                min_value=0.1,
                max_value=0.5,
                step=0.1,
                default=0.3
            )
            x = layers.Dropout(dropout_rate, name=f'dropout_conv_{i}')(x)
        
        # ===== LSTM HYPERPARAMETERS =====
        
        # Use LSTM or not
        use_lstm = hp.Boolean('use_lstm', default=True)
        
        if use_lstm:
            lstm_units = hp.Int('lstm_units', min_value=64, max_value=256, step=32, default=128)
            lstm_dropout = hp.Float('lstm_dropout', min_value=0.1, max_value=0.5, step=0.1, default=0.3)
            lstm_recurrent_dropout = hp.Float('lstm_recurrent_dropout', min_value=0.1, max_value=0.5, step=0.1, default=0.3)
            
            x = layers.Bidirectional(
                layers.LSTM(
                    lstm_units,
                    return_sequences=False,
                    dropout=lstm_dropout,
                    recurrent_dropout=lstm_recurrent_dropout,
                    name='lstm'
                ),
                name='bidirectional_lstm'
            )(x)
        else:
            # Use GlobalAveragePooling instead
            x = layers.GlobalAveragePooling1D()(x)
        
        # ===== DENSE LAYER HYPERPARAMETERS =====
        
        # Number of dense layers
        n_dense_layers = hp.Int('n_dense_layers', min_value=1, max_value=3, default=2)
        
        for i in range(n_dense_layers):
            dense_units = hp.Int(
                f'dense_units_{i}',
                min_value=64,
                max_value=512,
                step=64,
                default=256
            )
            
            x = layers.Dense(
                dense_units,
                activation='relu',
                kernel_regularizer=regularizers.l2(hp.Float(
                    f'dense_l2_{i}',
                    min_value=1e-4,
                    max_value=1e-2,
                    sampling='log',
                    default=1e-3
                )),
                name=f'dense_{i}'
            )(x)
            
            x = layers.BatchNormalization(name=f'bn_dense_{i}')(x)
            
            dense_dropout = hp.Float(
                f'dense_dropout_{i}',
                min_value=0.2,
                max_value=0.7,
                step=0.1,
                default=0.5
            )
            x = layers.Dropout(dense_dropout, name=f'dropout_dense_{i}')(x)
        
        # ===== HIERARCHICAL OUTPUTS =====
        
        valence_output = layers.Dense(2, activation='softmax', name='valence')(x)
        arousal_output = layers.Dense(2, activation='softmax', name='arousal')(x)
        emotion_output = layers.Dense(self.n_classes, activation='softmax', name='emotion')(x)
        
        # Create model
        model = keras.Model(
            inputs=inputs,
            outputs=[valence_output, arousal_output, emotion_output],
            name='tuned_emotion_model'
        )
        
        # ===== OPTIMIZER HYPERPARAMETERS =====
        
        learning_rate = hp.Float(
            'learning_rate',
            min_value=1e-5,
            max_value=1e-2,
            sampling='log',
            default=1e-3
        )
        
        optimizer_name = hp.Choice('optimizer', values=['adam', 'adamw', 'rmsprop'], default='adam')
        
        if optimizer_name == 'adam':
            opt = optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name == 'adamw':
            opt = optimizers.AdamW(learning_rate=learning_rate)
        else:
            opt = optimizers.RMSprop(learning_rate=learning_rate)
        
        # Compile
        model.compile(
            optimizer=opt,
            loss={
                'valence': 'categorical_crossentropy',
                'arousal': 'categorical_crossentropy',
                'emotion': 'categorical_crossentropy'
            },
            loss_weights={
                'valence': 0.2,
                'arousal': 0.2,
                'emotion': 0.6
            },
            metrics=['accuracy']
        )
        
        return model
    
    def tune(
        self,
        X_train: np.ndarray,
        y_train: Dict[str, np.ndarray],
        X_val: np.ndarray,
        y_val: Dict[str, np.ndarray],
        method: str = 'bayesian',
        max_trials: int = 50,
        executions_per_trial: int = 1
    ) -> kt.Tuner:
        """
        Run hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training labels (dict with 'valence', 'arousal', 'emotion')
            X_val: Validation features
            y_val: Validation labels
            method: 'bayesian', 'hyperband', or 'random'
            max_trials: Maximum number of trials
            executions_per_trial: Runs per trial (for averaging)
            
        Returns:
            Fitted tuner object
        """
        logger.info(f"🔍 Starting {method} hyperparameter search with {max_trials} trials...")
        
        # Choose tuning algorithm
        if method == 'bayesian':
            tuner = kt.BayesianOptimization(
                hypermodel=self.build_model,
                objective=kt.Objective('val_emotion_accuracy', direction='max'),
                max_trials=max_trials,
                executions_per_trial=executions_per_trial,
                directory=str(self.tuner_dir),
                project_name=self.project_name,
                overwrite=False
            )
        elif method == 'hyperband':
            tuner = kt.Hyperband(
                hypermodel=self.build_model,
                objective=kt.Objective('val_emotion_accuracy', direction='max'),
                max_epochs=50,
                factor=3,
                directory=str(self.tuner_dir),
                project_name=self.project_name,
                overwrite=False
            )
        else:  # random
            tuner = kt.RandomSearch(
                hypermodel=self.build_model,
                objective=kt.Objective('val_emotion_accuracy', direction='max'),
                max_trials=max_trials,
                executions_per_trial=executions_per_trial,
                directory=str(self.tuner_dir),
                project_name=self.project_name,
                overwrite=False
            )
        
        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_emotion_accuracy',
            patience=10,
            restore_best_weights=True
        )
        
        # Run search
        tuner.search(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            callbacks=[early_stop],
            verbose=1
        )
        
        return tuner
    
    def get_best_hyperparameters(self, tuner: kt.Tuner) -> Dict[str, Any]:
        """Extract best hyperparameters from tuner."""
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        
        # Convert to dictionary
        hp_dict = {}
        for param in best_hps.space:
            hp_dict[param.name] = best_hps.get(param.name)
        
        return hp_dict
    
    def save_best_config(self, tuner: kt.Tuner, output_path: Optional[Path] = None) -> None:
        """Save best hyperparameters to config file."""
        if output_path is None:
            output_path = MODEL_DIR / "best_hyperparameters.json"
        
        best_hps = self.get_best_hyperparameters(tuner)
        
        # Add metadata
        config = {
            'hyperparameters': best_hps,
            'metadata': {
                'tuning_method': type(tuner).__name__,
                'best_score': tuner.get_best_models(1)[0].evaluate(return_dict=True),
                'timestamp': datetime.now().isoformat(),
                'total_trials': len(tuner.oracle.trials)
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"✅ Best hyperparameters saved to {output_path}")


# =============================================================================
# 2. OPTUNA IMPLEMENTATION (MOST ADVANCED)
# =============================================================================

class OptunaOptimizer:
    """
    Hyperparameter optimization using Optuna.
    
    Features:
    - Tree-structured Parzen Estimator (TPE) - Most efficient
    - Automatic pruning of bad trials
    - Advanced visualizations
    - Better than random/grid search
    
    Advantages over Keras Tuner:
    - Faster convergence
    - Better parallelization
    - More flexible trial management
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        n_classes: int = EMOTION_CLASSES,
        study_name: str = "emotion_recognition_study"
    ):
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna not installed")
        
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.study_name = study_name
        self.study = None
        
    def objective(
        self,
        trial: optuna.Trial,
        X_train: np.ndarray,
        y_train: Dict[str, np.ndarray],
        X_val: np.ndarray,
        y_val: Dict[str, np.ndarray]
    ) -> float:
        """
        Objective function for Optuna trial.
        
        Args:
            trial: Optuna trial object
            X_train, y_train: Training data
            X_val, y_val: Validation data
            
        Returns:
            Validation accuracy (to maximize)
        """
        # Clear session to prevent memory leaks
        keras.backend.clear_session()
        
        # ===== SUGGEST HYPERPARAMETERS =====
        
        # CNN parameters
        n_cnn_blocks = trial.suggest_int('n_cnn_blocks', 2, 4)
        
        # LSTM parameters
        use_lstm = trial.suggest_categorical('use_lstm', [True, False])
        lstm_units = trial.suggest_int('lstm_units', 64, 256, step=32) if use_lstm else None
        
        # Dense parameters
        n_dense_layers = trial.suggest_int('n_dense_layers', 1, 3)
        
        # Optimizer
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'adamw', 'rmsprop'])
        
        # Batch size
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        
        # ===== BUILD MODEL =====
        
        inputs = layers.Input(shape=self.input_shape)
        
        if len(self.input_shape) == 1:
            x = layers.Reshape((self.input_shape[0], 1))(inputs)
        else:
            x = inputs
        
        # CNN blocks
        for i in range(n_cnn_blocks):
            filters = trial.suggest_int(f'cnn_filters_{i}', 32, 256, step=32)
            kernel_size = trial.suggest_categorical(f'kernel_size_{i}', [3, 5, 7])
            dropout = trial.suggest_float(f'cnn_dropout_{i}', 0.1, 0.5, step=0.1)
            
            x = layers.Conv1D(filters, kernel_size, padding='same', activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling1D(2)(x)
            x = layers.Dropout(dropout)(x)
        
        # LSTM or pooling
        if use_lstm:
            lstm_dropout = trial.suggest_float('lstm_dropout', 0.1, 0.5, step=0.1)
            x = layers.Bidirectional(
                layers.LSTM(lstm_units, return_sequences=False, dropout=lstm_dropout)
            )(x)
        else:
            x = layers.GlobalAveragePooling1D()(x)
        
        # Dense layers
        for i in range(n_dense_layers):
            units = trial.suggest_int(f'dense_units_{i}', 64, 512, step=64)
            dropout = trial.suggest_float(f'dense_dropout_{i}', 0.2, 0.7, step=0.1)
            
            x = layers.Dense(units, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(dropout)(x)
        
        # Outputs
        valence_output = layers.Dense(2, activation='softmax', name='valence')(x)
        arousal_output = layers.Dense(2, activation='softmax', name='arousal')(x)
        emotion_output = layers.Dense(self.n_classes, activation='softmax', name='emotion')(x)
        
        model = keras.Model(inputs=inputs, outputs=[valence_output, arousal_output, emotion_output])
        
        # Compile
        if optimizer_name == 'adam':
            opt = optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name == 'adamw':
            opt = optimizers.AdamW(learning_rate=learning_rate)
        else:
            opt = optimizers.RMSprop(learning_rate=learning_rate)
        
        model.compile(
            optimizer=opt,
            loss={
                'valence': 'categorical_crossentropy',
                'arousal': 'categorical_crossentropy',
                'emotion': 'categorical_crossentropy'
            },
            loss_weights={'valence': 0.2, 'arousal': 0.2, 'emotion': 0.6},
            metrics=['accuracy']
        )
        
        # ===== TRAIN MODEL =====
        
        # Pruning callback
        pruning_callback = TFKerasPruningCallback(trial, 'val_emotion_accuracy')
        
        early_stop = callbacks.EarlyStopping(
            monitor='val_emotion_accuracy',
            patience=10,
            restore_best_weights=True
        )
        
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=batch_size,
            callbacks=[pruning_callback, early_stop],
            verbose=0
        )
        
        # Return best validation accuracy
        return max(history.history['val_emotion_accuracy'])
    
    def optimize(
        self,
        X_train: np.ndarray,
        y_train: Dict[str, np.ndarray],
        X_val: np.ndarray,
        y_val: Dict[str, np.ndarray],
        n_trials: int = 100,
        timeout: Optional[int] = None
    ) -> optuna.Study:
        """
        Run Optuna optimization.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            n_trials: Number of trials
            timeout: Timeout in seconds (optional)
            
        Returns:
            Optuna study object
        """
        logger.info(f"🚀 Starting Optuna optimization with {n_trials} trials...")
        
        # Create study
        self.study = optuna.create_study(
            study_name=self.study_name,
            direction='maximize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Optimize
        self.study.optimize(
            lambda trial: self.objective(trial, X_train, y_train, X_val, y_val),
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        logger.info(f"✅ Optimization complete! Best accuracy: {self.study.best_value:.4f}")
        
        return self.study
    
    def save_best_config(self, output_path: Optional[Path] = None) -> None:
        """Save best hyperparameters."""
        if output_path is None:
            output_path = MODEL_DIR / "best_hyperparameters_optuna.json"
        
        config = {
            'hyperparameters': self.study.best_params,
            'metadata': {
                'best_score': self.study.best_value,
                'n_trials': len(self.study.trials),
                'timestamp': datetime.now().isoformat()
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"✅ Best hyperparameters saved to {output_path}")


# =============================================================================
# 3. GRID/RANDOM SEARCH (BASELINE)
# =============================================================================

class ManualSearchOptimizer:
    """
    Traditional grid/random search for baseline comparison.
    Simple but exhaustive.
    """
    
    def __init__(self, input_shape: Tuple[int, ...], n_classes: int = EMOTION_CLASSES):
        self.input_shape = input_shape
        self.n_classes = n_classes
        
    def grid_search(
        self,
        X_train: np.ndarray,
        y_train: Dict[str, np.ndarray],
        X_val: np.ndarray,
        y_val: Dict[str, np.ndarray],
        param_grid: Dict[str, List[Any]]
    ) -> Dict[str, Any]:
        """
        Exhaustive grid search over parameter grid.
        
        Warning: Can be very slow for large grids!
        """
        from itertools import product
        
        best_score = 0
        best_params = {}
        
        # Generate all combinations
        keys, values = zip(*param_grid.items())
        combinations = [dict(zip(keys, v)) for v in product(*values)]
        
        logger.info(f"Testing {len(combinations)} combinations...")
        
        for i, params in enumerate(combinations):
            logger.info(f"Trial {i+1}/{len(combinations)}: {params}")
            
            # Build and train model with these params
            # (simplified for demonstration)
            score = self._evaluate_params(params, X_train, y_train, X_val, y_val)
            
            if score > best_score:
                best_score = score
                best_params = params
                logger.info(f"✨ New best score: {best_score:.4f}")
        
        return best_params
    
    def _evaluate_params(self, params, X_train, y_train, X_val, y_val):
        """Evaluate single parameter configuration."""
        # Implement model building and training with params
        # Return validation accuracy
        pass


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def prepare_data_for_tuning(
    X: np.ndarray,
    y_labels: np.ndarray
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Prepare data in format expected by hierarchical model.
    
    Args:
        X: Features
        y_labels: Emotion labels (integers 0-4)
        
    Returns:
        X, Dict with valence/arousal/emotion labels
    """
    from tensorflow.keras.utils import to_categorical
    
    # Valence/arousal mapping
    valence_map = {0: 0, 1: 1, 2: 0, 3: 1, 4: 1}
    arousal_map = {0: 0, 1: 1, 2: 0, 3: 0, 4: 1}
    
    valence = np.array([valence_map[int(y)] for y in y_labels])
    arousal = np.array([arousal_map[int(y)] for y in y_labels])
    
    y_dict = {
        'valence': to_categorical(valence, 2),
        'arousal': to_categorical(arousal, 2),
        'emotion': to_categorical(y_labels, EMOTION_CLASSES)
    }
    
    return X, y_dict


def load_best_hyperparameters(config_path: Path) -> Dict[str, Any]:
    """Load best hyperparameters from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config['hyperparameters']


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Run hyperparameter tuning from command line."""
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for emotion recognition')
    parser.add_argument('--method', type=str, default='keras-tuner',
                       choices=['keras-tuner', 'optuna', 'grid-search'],
                       help='Optimization method')
    parser.add_argument('--max-trials', type=int, default=50,
                       help='Maximum number of trials')
    parser.add_argument('--tuning-algorithm', type=str, default='bayesian',
                       choices=['bayesian', 'hyperband', 'random'],
                       help='Keras Tuner algorithm (if method=keras-tuner)')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to preprocessed data (X_train.npy, y_train.npy, etc.)')
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data from {args.data_path}...")
    data_path = Path(args.data_path)
    X_train = np.load(data_path / 'X_train.npy')
    y_train = np.load(data_path / 'y_train.npy')
    X_val = np.load(data_path / 'X_val.npy')
    y_val = np.load(data_path / 'y_val.npy')
    
    # Prepare hierarchical labels
    _, y_train_dict = prepare_data_for_tuning(X_train, y_train)
    _, y_val_dict = prepare_data_for_tuning(X_val, y_val)
    
    # Run tuning
    if args.method == 'keras-tuner':
        optimizer = KerasTunerOptimizer(input_shape=(X_train.shape[1],))
        tuner = optimizer.tune(
            X_train, y_train_dict,
            X_val, y_val_dict,
            method=args.tuning_algorithm,
            max_trials=args.max_trials
        )
        optimizer.save_best_config(tuner)
        
    elif args.method == 'optuna':
        optimizer = OptunaOptimizer(input_shape=(X_train.shape[1],))
        study = optimizer.optimize(
            X_train, y_train_dict,
            X_val, y_val_dict,
            n_trials=args.max_trials
        )
        optimizer.save_best_config()
    
    logger.info("🎉 Hyperparameter tuning complete!")


if __name__ == "__main__":
    main()
