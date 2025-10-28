# Deep Learning Emotion Recognition Module
# Integrates CNN+BiLSTM hybrid model with frontal alpha asymmetry for enhanced emotion detection
# Based on Frantzidis et al. (2010) and modern deep learning approaches for EEG emotion recognition

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import os

# Optional TensorFlow/Keras imports (only if deep learning mode is enabled)
try:
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Bidirectional, Flatten, Dense, Dropout
    from tensorflow.keras.utils import to_categorical
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("TensorFlow/Keras not available. Deep learning features disabled.")

from app_configuration import FS, BANDPASS_LOW, BANDPASS_HIGH, BUTTER_ORDER

class DeepLearningEmotionRecognizer:
    """
    Enhanced emotion recognition using deep learning (CNN+BiLSTM) with:
    - Bandpower features across 5 frequency bands (delta, theta, alpha, beta, gamma)
    - Frontal Alpha Asymmetry (FAA) for improved accuracy
    - Hierarchical valence/arousal classification
    - Support for both training mode and inference mode
    """
    
    def __init__(self, use_deep_learning=False, model_path=None):
        """
        Initialize the emotion recognizer.
        
        Args:
            use_deep_learning: If True, uses CNN+BiLSTM; if False, falls back to threshold-based
            model_path: Path to load pre-trained model
        """
        self.fs = FS
        self.use_deep_learning = use_deep_learning and KERAS_AVAILABLE
        self.model = None
        self.scaler = None
        self.label_encoder = None
        
        # EEG frequency bands (Week 2: Band power analysis)
        self.bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }
        
        # Design bandpass filter (Week 3: Signal filtering)
        self.b, self.a = butter(BUTTER_ORDER, [BANDPASS_LOW/(0.5*self.fs), BANDPASS_HIGH/(0.5*self.fs)], btype='band')
        
        # Load pre-trained model if available
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def bandpass_filter(self, signal):
        """Apply bandpass filter to remove noise (Week 3: Convolution theory)"""
        return filtfilt(self.b, self.a, signal)
    
    def extract_bandpower(self, signal, band):
        """
        Extract power in a specific frequency band using FFT (Week 2: Fourier analysis).
        
        Args:
            signal: 1D array of EEG data
            band: Tuple (low_freq, high_freq) in Hz
        
        Returns:
            Band power value
        """
        N = len(signal)
        yf = fft(signal)
        xf = fftfreq(N, 1/self.fs)[:N//2]
        power = (2.0/N) * np.abs(yf[0:N//2])**2
        idx = (xf >= band[0]) & (xf <= band[1])
        return np.sum(power[idx]) + 1e-6  # Small epsilon to avoid log(0)
    
    def extract_features(self, eeg_window):
        """
        Extract comprehensive features from EEG window including:
        - Bandpower for all 5 bands per channel
        - Frontal Alpha Asymmetry (FAA) if 2+ channels available
        
        Args:
            eeg_window: 2D array (n_channels, n_samples) or 1D array (n_samples)
        
        Returns:
            Feature vector as numpy array
        """
        # Handle single-channel case
        if eeg_window.ndim == 1:
            eeg_window = eeg_window.reshape(1, -1)
        
        feature_list = []
        
        # Extract bandpower for each channel (Week 2: Multi-band analysis)
        for ch_idx in range(eeg_window.shape[0]):
            ch_signal = self.bandpass_filter(eeg_window[ch_idx])
            ch_features = [self.extract_bandpower(ch_signal, band) for band in self.bands.values()]
            feature_list.extend(ch_features)
        
        # Frontal Alpha Asymmetry (FAA) - key for emotion detection (Frantzidis et al., 2010)
        # Assumes channel 0 = Fp1 (left frontal), channel 1 = Fp2 (right frontal)
        if eeg_window.shape[0] >= 2:
            left_alpha = self.extract_bandpower(self.bandpass_filter(eeg_window[0]), self.bands['alpha'])
            right_alpha = self.extract_bandpower(self.bandpass_filter(eeg_window[1]), self.bands['alpha'])
            faa = np.log(right_alpha) - np.log(left_alpha)  # Log ratio reduces happy/sad confusion
            feature_list.append(faa)
        
        return np.array(feature_list)
    
    def build_cnn_lstm_model(self, input_shape, num_classes):
        """
        Build hybrid CNN+BiLSTM model for temporal emotion recognition.
        
        Architecture:
        - Conv1D layers: Extract spatial-frequency patterns (Week 4: Time-frequency analysis)
        - BiLSTM: Capture temporal dynamics
        - Dense layers: Hierarchical classification
        
        Args:
            input_shape: Tuple (n_features, 1) for CNN input
            num_classes: Number of emotion classes
        
        Returns:
            Compiled Keras model
        """
        if not KERAS_AVAILABLE:
            raise RuntimeError("TensorFlow/Keras required for deep learning mode")
        
        inp = Input(shape=input_shape)
        
        # CNN layers for feature extraction
        x = Conv1D(64, 3, activation='relu', padding='same')(inp)
        x = MaxPooling1D(2)(x)
        x = Conv1D(128, 3, activation='relu', padding='same')(x)
        x = MaxPooling1D(2)(x)
        
        # BiLSTM for temporal context
        x = Bidirectional(LSTM(64, return_sequences=True))(x)
        x = Flatten()(x)
        
        # Dense layers for classification
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        out = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inp, out)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        return model
    
    def hierarchical_classify(self, features):
        """
        Hierarchical valence/arousal classification (Frantzidis approach).
        Falls back to threshold-based if deep learning unavailable.
        
        Args:
            features: Feature vector
        
        Returns:
            Emotion label string
        """
        # Extract key ratios
        if len(features) >= 10:  # Multi-channel with FAA
            alpha_power = features[2]  # Alpha is 3rd band
            beta_power = features[3]
            theta_power = features[1]
            faa = features[-1] if len(features) > 10 else 0
        else:  # Single channel
            alpha_power = features[2] if len(features) > 2 else 1
            beta_power = features[3] if len(features) > 3 else 1
            theta_power = features[1] if len(features) > 1 else 1
            faa = 0
        
        beta_alpha = beta_power / (alpha_power + 1e-6)
        alpha_theta = alpha_power / (theta_power + 1e-6)
        
        # Hierarchical decision tree (Week 4: Multi-class classification)
        if beta_alpha > 1.5 and alpha_theta > 1.0:
            return "focus"  # High beta, moderate alpha
        elif alpha_theta > 1.2 and beta_alpha < 1.0:
            return "relax"  # High alpha, low beta
        elif faa > 0.2 or (alpha_power > beta_power and theta_power < alpha_power):
            return "happy"  # Right frontal activation or positive valence
        elif faa < -0.2 or (theta_power > alpha_power):
            return "sad"  # Left frontal activation or negative valence
        else:
            return "fatigue"  # Low overall activity
    
    def predict_emotion(self, eeg_window):
        """
        Predict emotion from EEG window using either deep learning or hierarchical rules.
        
        Args:
            eeg_window: 2D array (n_channels, n_samples) or 1D array (n_samples)
        
        Returns:
            Tuple (emotion_label, confidence, features_dict)
        """
        # Extract features
        features = self.extract_features(eeg_window)
        
        # Deep learning inference if available
        if self.use_deep_learning and self.model is not None:
            X_scaled = self.scaler.transform(features.reshape(1, -1))
            X_cnn = X_scaled.reshape(1, -1, 1)
            pred = self.model.predict(X_cnn, verbose=0)
            emotion = self.label_encoder.inverse_transform([np.argmax(pred)])[0]
            confidence = float(np.max(pred))
        else:
            # Fallback to hierarchical classification
            emotion = self.hierarchical_classify(features)
            confidence = 0.75  # Estimated confidence for rule-based
        
        # Package features for logging
        features_dict = {
            'alpha_power': float(features[2]) if len(features) > 2 else 0,
            'beta_power': float(features[3]) if len(features) > 3 else 0,
            'theta_power': float(features[1]) if len(features) > 1 else 0,
            'delta_power': float(features[0]) if len(features) > 0 else 0,
            'gamma_power': float(features[4]) if len(features) > 4 else 0,
            'faa': float(features[-1]) if len(features) > 10 else 0
        }
        
        return emotion, confidence, features_dict
    
    def train_model(self, X_data, y_labels, epochs=20, batch_size=32, validation_split=0.2):
        """
        Train the deep learning model on EEG data.
        
        Args:
            X_data: Array of feature vectors or raw EEG windows
            y_labels: Array of emotion labels
            epochs: Training epochs
            batch_size: Batch size
            validation_split: Validation data fraction
        
        Returns:
            Training history
        """
        if not KERAS_AVAILABLE:
            print("TensorFlow/Keras not available. Cannot train deep learning model.")
            return None
        
        # Process labels
        self.label_encoder = LabelEncoder()
        y_enc = self.label_encoder.fit_transform(y_labels)
        y_cat = to_categorical(y_enc)
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_data)
        X_cnn = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
        
        # Build model
        self.model = self.build_cnn_lstm_model((X_cnn.shape[1], 1), num_classes=y_cat.shape[1])
        
        # Train
        history = self.model.fit(
            X_cnn, y_cat,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        self.use_deep_learning = True
        return history
    
    def save_model(self, path):
        """Save trained model, scaler, and label encoder"""
        if self.model is None:
            print("No model to save")
            return
        
        self.model.save(f"{path}_model.h5")
        with open(f"{path}_scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(f"{path}_encoder.pkl", 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load pre-trained model, scaler, and label encoder"""
        if not KERAS_AVAILABLE:
            print("TensorFlow/Keras not available. Cannot load deep learning model.")
            return
        
        try:
            self.model = load_model(f"{path}_model.h5")
            with open(f"{path}_scaler.pkl", 'rb') as f:
                self.scaler = pickle.load(f)
            with open(f"{path}_encoder.pkl", 'rb') as f:
                self.label_encoder = pickle.load(f)
            self.use_deep_learning = True
            print(f"Model loaded from {path}")
        except Exception as e:
            print(f"Error loading model: {e}")
