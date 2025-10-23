"""
Emotion Recognition Deep Learning Model
========================================

CNN+BiLSTM hybrid architecture for EEG-based emotion recognition with
hierarchical classification. Implements state-of-the-art techniques from:
- EEGNet (Lawhern et al., 2018): Compact CNN for EEG
- Li et al. (2018): LSTM for temporal EEG patterns
- Zheng & Lu (2015): Deep learning for emotion recognition
- Yang et al. (2018): Hierarchical emotion classification

Architecture:
- Convolutional layers: Extract spatial-frequency features
- Bidirectional LSTM: Capture temporal dynamics
- Hierarchical output: Valence (binary) + Arousal (binary) + Emotion (multi-class)
- Regularization: Dropout, batch normalization, L2 weight decay

Author: Rebuilt for CMP9780M Assessment
License: Proprietary (see root LICENSE)
"""

import numpy as np
from typing import Tuple, Optional, List, Dict, Union
import logging
import pickle
from pathlib import Path

# TensorFlow/Keras imports with graceful fallback
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks, optimizers, regularizers
    from tensorflow.keras.utils import to_categorical
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("WARNING: TensorFlow not available. Model training/inference disabled.")

# Import configuration
import sys
sys.path.append(str(Path(__file__).parent))
from config import (
    SAMPLING_RATE, N_FEATURES, EMOTION_CLASSES, EMOTION_LABELS, EMOTION_TO_ID,
    VALENCE_CLASSES, AROUSAL_CLASSES, CNN_FILTERS, CNN_KERNEL_SIZE, CNN_POOL_SIZE,
    CNN_DROPOUT, LSTM_UNITS, LSTM_DROPOUT, LSTM_RECURRENT_DROPOUT,
    DENSE_UNITS, DENSE_DROPOUT, LEARNING_RATE, OPTIMIZER, BATCH_SIZE, EPOCHS,
    VALIDATION_SPLIT, EARLY_STOPPING_PATIENCE, EARLY_STOPPING_MIN_DELTA,
    REDUCE_LR_PATIENCE, REDUCE_LR_FACTOR, REDUCE_LR_MIN_LR,
    CHECKPOINT_MONITOR, CHECKPOINT_MODE, MODEL_DIR
)

# Configure logging
logger = logging.getLogger(__name__)


class EmotionRecognitionModel:
    """
    Hybrid CNN+BiLSTM model for EEG-based emotion recognition.
    
    Features:
    - Hierarchical classification (valence, arousal, emotion)
    - Spatial feature extraction via CNN
    - Temporal modeling via BiLSTM
    - Comprehensive regularization
    - Early stopping and learning rate scheduling
    - Model checkpointing
    
    Attributes:
        model: Keras model instance
        label_encoder: LabelEncoder for emotion labels
        history: Training history
        
    Example:
        >>> model = EmotionRecognitionModel(input_shape=(n_features,))
        >>> model.build_model()
        >>> model.train(X_train, y_train, X_val, y_val)
        >>> predictions = model.predict(X_test)
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        n_classes: int = EMOTION_CLASSES,
        model_name: str = "emotion_cnn_bilstm"
    ):
        """
        Initialize emotion recognition model.
        
        Args:
            input_shape: Shape of input features (n_features,) or (n_timesteps, n_features)
            n_classes: Number of emotion classes
            model_name: Name for model saving/loading
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError(
                "TensorFlow not installed. Install with: pip install tensorflow>=2.10.0"
            )
        
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.model_name = model_name
        self.model = None
        self.label_encoder = LabelEncoder()
        self.history = None
        
        logger.info(f"Initialized {model_name} with input shape {input_shape}, {n_classes} classes")
    
    def build_model(
        self,
        architecture: str = 'cnn_bilstm'
    ) -> keras.Model:
        """
        Build the deep learning model architecture.
        
        Supports multiple architectures:
        - 'cnn_bilstm': Hybrid CNN+BiLSTM (recommended)
        - 'cnn': CNN-only (faster inference)
        - 'bilstm': BiLSTM-only (pure temporal)
        - 'dense': Simple MLP (baseline)
        
        Args:
            architecture: Model architecture type
            
        Returns:
            keras.Model: Compiled model ready for training
        """
        if architecture == 'cnn_bilstm':
            self.model = self._build_cnn_bilstm_model()
        elif architecture == 'cnn':
            self.model = self._build_cnn_model()
        elif architecture == 'bilstm':
            self.model = self._build_bilstm_model()
        elif architecture == 'dense':
            self.model = self._build_dense_model()
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        logger.info(f"Built {architecture} model")
        return self.model
    
    def _build_cnn_bilstm_model(self) -> keras.Model:
        """
        Build hybrid CNN+BiLSTM model for spatial-temporal feature extraction.
        
        Architecture:
        1. Input reshape for CNN compatibility
        2. Conv1D layers (64, 128, 256 filters) - Extract spatial-frequency patterns
        3. MaxPooling1D - Reduce dimensionality
        4. Bidirectional LSTM (128 units) - Capture temporal dependencies
        5. Dense layers (256, 128) - High-level feature combination
        6. Hierarchical outputs:
           - Valence (2 classes): Positive/Negative
           - Arousal (2 classes): High/Low
           - Emotion (5 classes): Happy/Sad/Relaxed/Focused/Neutral
        
        Returns:
            keras.Model: Compiled hierarchical model
        """
        # Input layer
        inputs = layers.Input(shape=self.input_shape, name='feature_input')
        
        # Reshape for CNN (add channel dimension if needed)
        if len(self.input_shape) == 1:
            # (n_features,) -> (n_features, 1) for Conv1D
            x = layers.Reshape((self.input_shape[0], 1))(inputs)
        else:
            x = inputs
        
        # CNN Block 1: Initial feature extraction
        x = layers.Conv1D(
            filters=CNN_FILTERS[0],
            kernel_size=CNN_KERNEL_SIZE,
            padding='same',
            activation='relu',
            kernel_regularizer=regularizers.l2(0.001),
            name='conv1d_1'
        )(x)
        x = layers.BatchNormalization(name='bn_1')(x)
        x = layers.MaxPooling1D(pool_size=CNN_POOL_SIZE, name='maxpool_1')(x)
        x = layers.Dropout(CNN_DROPOUT, name='dropout_conv_1')(x)
        
        # CNN Block 2: Deep feature extraction
        x = layers.Conv1D(
            filters=CNN_FILTERS[1],
            kernel_size=CNN_KERNEL_SIZE,
            padding='same',
            activation='relu',
            kernel_regularizer=regularizers.l2(0.001),
            name='conv1d_2'
        )(x)
        x = layers.BatchNormalization(name='bn_2')(x)
        x = layers.MaxPooling1D(pool_size=CNN_POOL_SIZE, name='maxpool_2')(x)
        x = layers.Dropout(CNN_DROPOUT, name='dropout_conv_2')(x)
        
        # CNN Block 3: High-level feature extraction
        x = layers.Conv1D(
            filters=CNN_FILTERS[2],
            kernel_size=CNN_KERNEL_SIZE,
            padding='same',
            activation='relu',
            kernel_regularizer=regularizers.l2(0.001),
            name='conv1d_3'
        )(x)
        x = layers.BatchNormalization(name='bn_3')(x)
        x = layers.Dropout(CNN_DROPOUT, name='dropout_conv_3')(x)
        
        # Bidirectional LSTM: Temporal modeling
        x = layers.Bidirectional(
            layers.LSTM(
                LSTM_UNITS,
                return_sequences=False,  # Only last output
                dropout=LSTM_DROPOUT,
                recurrent_dropout=LSTM_RECURRENT_DROPOUT,
                name='lstm'
            ),
            name='bidirectional_lstm'
        )(x)
        
        # Dense Block: High-level feature combination
        x = layers.Dense(
            DENSE_UNITS[0],
            activation='relu',
            kernel_regularizer=regularizers.l2(0.001),
            name='dense_1'
        )(x)
        x = layers.BatchNormalization(name='bn_dense_1')(x)
        x = layers.Dropout(DENSE_DROPOUT, name='dropout_dense_1')(x)
        
        x = layers.Dense(
            DENSE_UNITS[1],
            activation='relu',
            kernel_regularizer=regularizers.l2(0.001),
            name='dense_2'
        )(x)
        x = layers.BatchNormalization(name='bn_dense_2')(x)
        x = layers.Dropout(DENSE_DROPOUT, name='dropout_dense_2')(x)
        
        # Hierarchical outputs
        valence_output = layers.Dense(
            VALENCE_CLASSES,
            activation='softmax',
            name='valence'
        )(x)
        
        arousal_output = layers.Dense(
            AROUSAL_CLASSES,
            activation='softmax',
            name='arousal'
        )(x)
        
        emotion_output = layers.Dense(
            self.n_classes,
            activation='softmax',
            name='emotion'
        )(x)
        
        # Create model with multiple outputs
        model = keras.Model(
            inputs=inputs,
            outputs=[valence_output, arousal_output, emotion_output],
            name='hierarchical_emotion_model'
        )
        
        # Compile with different loss weights
        model.compile(
            optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
            loss={
                'valence': 'categorical_crossentropy',
                'arousal': 'categorical_crossentropy',
                'emotion': 'categorical_crossentropy'
            },
            loss_weights={
                'valence': 0.2,  # Lower weight (auxiliary task)
                'arousal': 0.2,  # Lower weight (auxiliary task)
                'emotion': 0.6   # Higher weight (primary task)
            },
            metrics=['accuracy']
        )
        
        return model
    
    def _build_cnn_model(self) -> keras.Model:
        """Build CNN-only model (faster inference, no temporal modeling)."""
        inputs = layers.Input(shape=self.input_shape)
        
        if len(self.input_shape) == 1:
            x = layers.Reshape((self.input_shape[0], 1))(inputs)
        else:
            x = inputs
        
        # CNN blocks
        for i, filters in enumerate(CNN_FILTERS):
            x = layers.Conv1D(
                filters=filters,
                kernel_size=CNN_KERNEL_SIZE,
                padding='same',
                activation='relu',
                kernel_regularizer=regularizers.l2(0.001)
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling1D(pool_size=CNN_POOL_SIZE)(x)
            x = layers.Dropout(CNN_DROPOUT)(x)
        
        # Global pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Dense layers
        for units in DENSE_UNITS:
            x = layers.Dense(units, activation='relu',
                           kernel_regularizer=regularizers.l2(0.001))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(DENSE_DROPOUT)(x)
        
        # Output
        outputs = layers.Dense(self.n_classes, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='cnn_emotion_model')
        model.compile(
            optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _build_bilstm_model(self) -> keras.Model:
        """Build BiLSTM-only model (pure temporal modeling)."""
        inputs = layers.Input(shape=self.input_shape)
        
        # Ensure 3D input for LSTM
        if len(self.input_shape) == 1:
            x = layers.Reshape((1, self.input_shape[0]))(inputs)
        else:
            x = inputs
        
        # BiLSTM layers
        x = layers.Bidirectional(
            layers.LSTM(LSTM_UNITS, return_sequences=True,
                       dropout=LSTM_DROPOUT, recurrent_dropout=LSTM_RECURRENT_DROPOUT)
        )(x)
        
        x = layers.Bidirectional(
            layers.LSTM(LSTM_UNITS // 2, return_sequences=False,
                       dropout=LSTM_DROPOUT, recurrent_dropout=LSTM_RECURRENT_DROPOUT)
        )(x)
        
        # Dense layers
        for units in DENSE_UNITS:
            x = layers.Dense(units, activation='relu',
                           kernel_regularizer=regularizers.l2(0.001))(x)
            x = layers.Dropout(DENSE_DROPOUT)(x)
        
        outputs = layers.Dense(self.n_classes, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='bilstm_emotion_model')
        model.compile(
            optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _build_dense_model(self) -> keras.Model:
        """Build simple MLP as baseline."""
        inputs = layers.Input(shape=self.input_shape)
        
        # Flatten if needed
        if len(self.input_shape) > 1:
            x = layers.Flatten()(inputs)
        else:
            x = inputs
        
        # Dense layers
        for units in DENSE_UNITS + [self.n_classes]:
            x = layers.Dense(units, activation='relu' if units != self.n_classes else 'softmax',
                           kernel_regularizer=regularizers.l2(0.001))(x)
            if units != self.n_classes:
                x = layers.Dropout(DENSE_DROPOUT)(x)
        
        model = keras.Model(inputs=inputs, outputs=x, name='dense_emotion_model')
        model.compile(
            optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def get_callbacks(
        self,
        checkpoint_path: Optional[str] = None
    ) -> List[callbacks.Callback]:
        """
        Create training callbacks for optimization and monitoring.
        
        Callbacks:
        - EarlyStopping: Stop if validation loss doesn't improve
        - ReduceLROnPlateau: Lower learning rate when stuck
        - ModelCheckpoint: Save best model
        - TensorBoard: Log for visualization (optional)
        
        Args:
            checkpoint_path: Path to save model checkpoints
            
        Returns:
            List of Keras callbacks
        """
        callback_list = []
        
        # Early stopping
        early_stop = callbacks.EarlyStopping(
            monitor=EARLY_STOPPING_MONITOR,
            patience=EARLY_STOPPING_PATIENCE,
            min_delta=EARLY_STOPPING_MIN_DELTA,
            restore_best_weights=True,
            verbose=1
        )
        callback_list.append(early_stop)
        
        # Learning rate reduction
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor=EARLY_STOPPING_MONITOR,
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=REDUCE_LR_MIN_LR,
            verbose=1
        )
        callback_list.append(reduce_lr)
        
        # Model checkpointing
        if checkpoint_path is None:
            checkpoint_path = MODEL_DIR / f"{self.model_name}_best.h5"
        
        checkpoint = callbacks.ModelCheckpoint(
            str(checkpoint_path),
            monitor=CHECKPOINT_MONITOR,
            mode=CHECKPOINT_MODE,
            save_best_only=True,
            verbose=1
        )
        callback_list.append(checkpoint)
        
        # TensorBoard logging (optional)
        # tensorboard = callbacks.TensorBoard(
        #     log_dir=str(MODEL_DIR / 'logs'),
        #     histogram_freq=1
        # )
        # callback_list.append(tensorboard)
        
        return callback_list
    
    def prepare_labels(
        self,
        labels: Union[np.ndarray, List],
        fit_encoder: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare emotion labels for hierarchical classification.
        
        Converts emotion labels to:
        1. Valence labels (positive/negative)
        2. Arousal labels (high/low)
        3. Emotion labels (one-hot encoded)
        
        Args:
            labels: String labels ('happy', 'sad', etc.) or integer indices
            fit_encoder: Fit label encoder (True for training, False for inference)
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 
                (valence_labels, arousal_labels, emotion_labels)
        """
        # Convert to numpy array
        labels = np.array(labels)
        
        # Encode string labels to integers if needed
        if labels.dtype == object or labels.dtype.kind == 'U':
            if fit_encoder:
                emotion_indices = self.label_encoder.fit_transform(labels)
            else:
                emotion_indices = self.label_encoder.transform(labels)
        else:
            emotion_indices = labels
        
        # Emotion-to-valence/arousal mapping
        # Based on circumplex model of affect (Russell, 1980)
        valence_map = {
            0: 0,  # neutral -> neutral
            1: 1,  # happy -> positive
            2: 0,  # sad -> negative
            3: 1,  # relaxed -> positive
            4: 1,  # focused -> positive (slight)
        }
        
        arousal_map = {
            0: 0,  # neutral -> low
            1: 1,  # happy -> high
            2: 0,  # sad -> low
            3: 0,  # relaxed -> low
            4: 1,  # focused -> high
        }
        
        # Create valence and arousal labels
        valence_labels = np.array([valence_map.get(idx, 0) for idx in emotion_indices])
        arousal_labels = np.array([arousal_map.get(idx, 0) for idx in emotion_indices])
        
        # One-hot encode
        valence_onehot = to_categorical(valence_labels, VALENCE_CLASSES)
        arousal_onehot = to_categorical(arousal_labels, AROUSAL_CLASSES)
        emotion_onehot = to_categorical(emotion_indices, self.n_classes)
        
        return valence_onehot, arousal_onehot, emotion_onehot
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: Union[np.ndarray, List],
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[Union[np.ndarray, List]] = None,
        epochs: int = EPOCHS,
        batch_size: int = BATCH_SIZE,
        validation_split: float = VALIDATION_SPLIT,
        verbose: int = 1
    ) -> keras.callbacks.History:
        """
        Train the emotion recognition model.
        
        Args:
            X_train: Training features of shape (n_samples, n_features)
            y_train: Training labels (string or integer)
            X_val: Validation features (optional, will split from training if None)
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of training data to use for validation (if X_val=None)
            verbose: Verbosity level (0=silent, 1=progress, 2=one line per epoch)
            
        Returns:
            keras.callbacks.History: Training history
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Prepare hierarchical labels
        valence_train, arousal_train, emotion_train = self.prepare_labels(y_train, fit_encoder=True)
        
        # Prepare validation data
        if X_val is not None and y_val is not None:
            valence_val, arousal_val, emotion_val = self.prepare_labels(y_val, fit_encoder=False)
            validation_data = (
                X_val,
                {'valence': valence_val, 'arousal': arousal_val, 'emotion': emotion_val}
            )
            validation_split_actual = 0.0
        else:
            validation_data = None
            validation_split_actual = validation_split
        
        # Get callbacks
        callback_list = self.get_callbacks()
        
        # Train model
        logger.info(f"Training {self.model_name} for {epochs} epochs...")
        
        self.history = self.model.fit(
            X_train,
            {'valence': valence_train, 'arousal': arousal_train, 'emotion': emotion_train},
            validation_data=validation_data,
            validation_split=validation_split_actual,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=verbose
        )
        
        logger.info("Training complete!")
        
        return self.history
    
    def predict(
        self,
        X: np.ndarray,
        return_probs: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """
        Predict emotions from features.
        
        Args:
            X: Features of shape (n_samples, n_features)
            return_probs: Return probability distributions
            
        Returns:
            If return_probs=False:
                np.ndarray: Predicted emotion labels
            If return_probs=True:
                Tuple[np.ndarray, Dict]: (predicted_labels, probability_dict)
        """
        if self.model is None:
            raise ValueError("Model not built or loaded.")
        
        # Predict
        predictions = self.model.predict(X, verbose=0)
        
        # Handle different model architectures
        if isinstance(predictions, list):
            # Hierarchical model: [valence, arousal, emotion]
            valence_probs, arousal_probs, emotion_probs = predictions
        else:
            # Single-output model
            emotion_probs = predictions
            valence_probs = None
            arousal_probs = None
        
        # Get emotion class indices
        emotion_indices = np.argmax(emotion_probs, axis=1)
        
        # Convert to labels
        emotion_labels = self.label_encoder.inverse_transform(emotion_indices)
        
        if return_probs:
            probs_dict = {
                'emotion': emotion_probs,
                'valence': valence_probs,
                'arousal': arousal_probs
            }
            return emotion_labels, probs_dict
        
        return emotion_labels
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: Union[np.ndarray, List],
        verbose: int = 1
    ) -> Dict:
        """
        Evaluate model performance on test set.
        
        Args:
            X_test: Test features
            y_test: Test labels
            verbose: Verbosity level
            
        Returns:
            Dict containing:
                - 'accuracy': Overall accuracy
                - 'classification_report': Detailed metrics
                - 'confusion_matrix': Confusion matrix
        """
        # Predict
        y_pred = self.predict(X_test)
        
        # Convert y_test to same format as predictions
        if isinstance(y_test[0], (int, np.integer)):
            y_test_labels = self.label_encoder.inverse_transform(y_test)
        else:
            y_test_labels = y_test
        
        # Calculate metrics
        accuracy = np.mean(y_pred == y_test_labels)
        
        report = classification_report(
            y_test_labels,
            y_pred,
            target_names=[str(label) for label in self.label_encoder.classes_],
            output_dict=True
        )
        
        conf_matrix = confusion_matrix(y_test_labels, y_pred)
        
        if verbose:
            print("=" * 60)
            print("MODEL EVALUATION")
            print("=" * 60)
            print(f"Accuracy: {accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test_labels, y_pred))
            print("\nConfusion Matrix:")
            print(conf_matrix)
            print("=" * 60)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix
        }
    
    def save_model(
        self,
        path: Optional[Union[str, Path]] = None,
        save_weights_only: bool = False
    ) -> None:
        """
        Save trained model to disk.
        
        Args:
            path: Save path (defaults to MODEL_DIR/model_name)
            save_weights_only: Save only weights (not architecture)
        """
        if path is None:
            path = MODEL_DIR / f"{self.model_name}.h5"
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if save_weights_only:
            self.model.save_weights(str(path))
        else:
            self.model.save(str(path))
        
        # Save label encoder
        encoder_path = path.parent / f"{path.stem}_encoder.pkl"
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(
        self,
        path: Union[str, Path],
        load_weights_only: bool = False
    ) -> None:
        """
        Load trained model from disk.
        
        Args:
            path: Model file path
            load_weights_only: Load only weights (architecture must be built first)
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        if load_weights_only:
            if self.model is None:
                raise ValueError("Model architecture must be built before loading weights")
            self.model.load_weights(str(path))
        else:
            self.model = keras.models.load_model(str(path))
        
        # Load label encoder
        encoder_path = path.parent / f"{path.stem}_encoder.pkl"
        if encoder_path.exists():
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
        else:
            logger.warning(f"Label encoder not found at {encoder_path}")
        
        logger.info(f"Model loaded from {path}")
    
    def summary(self) -> None:
        """Print model architecture summary."""
        if self.model is None:
            print("Model not built yet.")
        else:
            self.model.summary()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_simple_emotion_model(
    n_features: int = N_FEATURES,
    n_classes: int = EMOTION_CLASSES
) -> EmotionRecognitionModel:
    """
    Create a simple emotion recognition model with default parameters.
    
    Convenience function for quick model creation.
    
    Args:
        n_features: Number of input features
        n_classes: Number of emotion classes
        
    Returns:
        EmotionRecognitionModel: Ready-to-train model
        
    Example:
        >>> model = create_simple_emotion_model(n_features=167, n_classes=5)
        >>> model.build_model(architecture='cnn_bilstm')
        >>> model.train(X_train, y_train)
    """
    model = EmotionRecognitionModel(
        input_shape=(n_features,),
        n_classes=n_classes,
        model_name='simple_emotion_model'
    )
    model.build_model(architecture='cnn_bilstm')
    return model


if __name__ == "__main__":
    # Demo and self-test
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow not available. Skipping model tests.")
    else:
        print("Emotion Recognition Model - Self Test")
        print("=" * 60)
        
        # Create dummy data
        np.random.seed(42)
        n_samples = 1000
        n_features = 167  # Example: 32 channels × 5 bands + 3 FAA
        n_classes = 5
        
        X_train = np.random.randn(n_samples, n_features).astype(np.float32)
        y_train = np.random.randint(0, n_classes, n_samples)
        
        X_test = np.random.randn(200, n_features).astype(np.float32)
        y_test = np.random.randint(0, n_classes, 200)
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Training labels shape: {y_train.shape}")
        
        # Create model
        print("\n--- Building Model ---")
        model = EmotionRecognitionModel(
            input_shape=(n_features,),
            n_classes=n_classes,
            model_name='test_model'
        )
        model.build_model(architecture='cnn_bilstm')
        model.summary()
        
        # Train for 2 epochs (quick test)
        print("\n--- Training Model (2 epochs for testing) ---")
        model.train(X_train, y_train, epochs=2, verbose=1)
        
        # Evaluate
        print("\n--- Evaluating Model ---")
        results = model.evaluate(X_test, y_test)
        
        # Save/load test
        print("\n--- Testing Save/Load ---")
        test_path = MODEL_DIR / "test_model.h5"
        model.save_model(test_path)
        
        model2 = EmotionRecognitionModel(
            input_shape=(n_features,),
            n_classes=n_classes
        )
        model2.load_model(test_path)
        print("Model loaded successfully!")
        
        # Clean up
        if test_path.exists():
            test_path.unlink()
        
        print("\nSelf-test complete!")
