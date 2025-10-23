# Signal processor module for EEG analysis and brain state classification
# Applies real-time filtering and feature extraction from the assessment (Weeks 2-4)
# Enhanced with optional deep learning support (CNN+BiLSTM) based on Frantzidis et al. (2010)

import numpy as np  # For arrays and math
from scipy.signal import butter, filtfilt, welch  # For filtering and PSD
from app_configuration import BANDPASS_LOW, BANDPASS_HIGH, BUTTER_ORDER, FS, FOCUS_THRESHOLD, RELAX_THRESHOLD, FATIGUE_THRESHOLD, HAPPY_THRESHOLD, SAD_THRESHOLD  # Import settings

# Try importing deep learning module
try:
    from dl_emotion_model import DeepLearningEmotionRecognizer
    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False
    print("Deep learning module not available. Using threshold-based classification.")

class SignalProcessor:
    # Initializes the processor with filter design and optional deep learning
    def __init__(self, use_deep_learning=False, model_path=None):
        """
        Args:
            use_deep_learning: If True and available, uses CNN+BiLSTM model
            model_path: Path to pre-trained deep learning model
        """
        self.fs = FS  # Sampling rate
        # Design Butterworth bandpass filter (Week 3: Convolution for EEG isolation)
        self.b, self.a = butter(BUTTER_ORDER, [BANDPASS_LOW, BANDPASS_HIGH], btype='band', fs=self.fs)
        
        # Initialize deep learning recognizer if available
        self.use_dl = use_deep_learning and DL_AVAILABLE
        if self.use_dl:
            self.dl_recognizer = DeepLearningEmotionRecognizer(use_deep_learning=True, model_path=model_path)
            print("Deep learning emotion recognition enabled (CNN+BiLSTM with Frontal Alpha Asymmetry)")
        else:
            self.dl_recognizer = None
            print("Using threshold-based emotion classification")

    # Processes a window of EEG data and classifies brain state
    def process_window(self, signal):
        """
        Process EEG window using either deep learning or threshold-based classification.
        
        Args:
            signal: 1D or 2D EEG data (n_samples) or (n_channels, n_samples)
        
        Returns:
            Tuple (state, features) where features is a dict
        """
        # Use deep learning if enabled
        if self.use_dl and self.dl_recognizer is not None:
            try:
                state, confidence, features = self.dl_recognizer.predict_emotion(signal)
                # Add confidence to features
                features['confidence'] = confidence
                features['method'] = 'deep_learning'
                return state, features
            except Exception as e:
                print(f"Deep learning error: {e}. Falling back to threshold-based.")
        
        # Fallback to original threshold-based method
        # Ensure signal is 1D for backward compatibility
        if signal.ndim > 1:
            signal = signal[0]  # Use first channel
        
        # Step 1: Apply bandpass filter to isolate 1-30 Hz EEG bands (Week 3: Noise attenuation)
        filtered = filtfilt(self.b, self.a, signal)  # Zero-phase filtering to avoid distortion
        
        # Step 2: Compute Power Spectral Density using Welch method (Week 2: Band estimation)
        try:
            f, Pxx = welch(filtered, fs=self.fs, nperseg=min(1024, len(filtered)))  # Segments for averaging, adjust for short signals
        except ValueError as e:
            print(f"Welch error: {e}. Using zeros.")
            f, Pxx = np.array([]), np.array([])
        
        # Step 3: Extract band powers (Week 2: Delta/Theta/Alpha/Beta analysis)
        alpha_band = (f >= 8) & (f <= 12)  # Alpha: 8-12 Hz
        beta_band = (f >= 12) & (f <= 30)  # Beta: 12-30 Hz
        theta_band = (f >= 4) & (f <= 8)   # Theta: 4-8 Hz
        
        alpha_power = np.mean(Pxx[alpha_band]) if np.any(alpha_band) and len(Pxx) > 0 else 0  # Mean power in band
        beta_power = np.mean(Pxx[beta_band]) if np.any(beta_band) and len(Pxx) > 0 else 0
        theta_power = np.mean(Pxx[theta_band]) if np.any(theta_band) and len(Pxx) > 0 else 0
        
        # Step 4: Calculate ratios for state indicators (Week 4: Feature-based detection, extended for emotions)
        beta_alpha = beta_power / alpha_power if alpha_power > 0 else 0  # Beta/Alpha for focus
        alpha_theta = alpha_power / theta_power if theta_power > 0 else 0  # Alpha/Theta for relax
        alpha_beta = alpha_power / beta_power if beta_power > 0 else 0  # Alpha/Beta for happiness
        theta_alpha = theta_power / alpha_power if alpha_power > 0 else 0  # Theta/Alpha for sadness
        
        # Step 5: Classify state using thresholds (Week 4: Threshold-based detection, extended for emotions)
        if beta_alpha > FOCUS_THRESHOLD:
            state = "focus"  # High beta/alpha
        elif alpha_theta > RELAX_THRESHOLD:
            state = "relax"  # High alpha/theta
        elif alpha_beta > HAPPY_THRESHOLD:
            state = "happy"  # High alpha/beta (positive arousal)
        elif theta_alpha > SAD_THRESHOLD:
            state = "sad"  # High theta/alpha (negative mood)
        else:
            state = "fatigue"  # Default low activity
        
        # Package features for logging
        features = {
            "alpha_power": alpha_power,
            "beta_power": beta_power,
            "theta_power": theta_power,
            "beta_alpha": beta_alpha,
            "alpha_theta": alpha_theta,
            "alpha_beta": alpha_beta,
            "theta_alpha": theta_alpha,
            "method": "threshold_based"
        }
        
        return state, features  # Return classified state and features