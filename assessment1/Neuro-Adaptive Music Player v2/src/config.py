"""
Configuration Module for Neuro-Adaptive Music Player v2
========================================================

Centralized configuration for all system parameters, hyperparameters, and paths.
Based on best practices from MNE-Python, EEGNet, and DEAP preprocessing pipelines.

Author: Rebuilt for CMP9780M Assessment
License: Proprietary (see root LICENSE)
"""

import os
from typing import Dict, List, Tuple
from pathlib import Path

# =============================================================================
# FILE PATHS AND DIRECTORIES
# =============================================================================

BASE_DIR = Path(__file__).parent.parent.resolve()
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"
MUSIC_DIR = BASE_DIR / "music"

# Create directories if they don't exist
for directory in [DATA_DIR, MODEL_DIR, LOG_DIR, MUSIC_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# =============================================================================
# EEG SIGNAL PARAMETERS (Based on international 10-20 system standards)
# =============================================================================

# Sampling parameters
SAMPLING_RATE: int = 256  # Hz - Standard clinical EEG rate (Nyquist-compliant for 0-128 Hz)
WINDOW_SIZE: float = 2.0  # seconds - Based on Frantzidis et al. (2010) optimal window
OVERLAP: float = 0.5  # 50% overlap for smoother transitions (Welch method standard)

# Channel configuration (10-20 system)
# Based on Frantzidis et al. frontal asymmetry research
FRONTAL_CHANNELS: List[str] = ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8']
TEMPORAL_CHANNELS: List[str] = ['T3', 'T4', 'T5', 'T6']
CENTRAL_CHANNELS: List[str] = ['C3', 'Cz', 'C4']
OCCIPITAL_CHANNELS: List[str] = ['O1', 'O2']

# Standard channel layout for DEAP/SEED datasets
STANDARD_CHANNEL_NAMES: List[str] = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
    'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
    'Fz', 'Cz', 'Pz', 'A1', 'A2'  # Include reference channels
]

# Minimum viable channel set for emotion recognition
MINIMAL_CHANNELS: List[str] = ['Fp1', 'Fp2']  # Sufficient for FAA

# =============================================================================
# SIGNAL PROCESSING PARAMETERS
# =============================================================================

# Bandpass filter parameters (Butterworth, as per Kothe & Makeig, 2013)
BANDPASS_LOW: float = 0.5  # Hz - Remove DC drift and very low frequencies
BANDPASS_HIGH: float = 45.0  # Hz - Remove line noise and high-frequency artifacts
FILTER_ORDER: int = 4  # 4th order provides good trade-off between sharpness and ripple

# Notch filter for powerline noise removal
NOTCH_FREQ: float = 50.0  # Hz - Europe/Asia (use 60.0 for North America)
NOTCH_Q: float = 30.0  # Quality factor - higher = narrower notch

# Artifact rejection thresholds (based on MNE-Python defaults)
VOLTAGE_THRESHOLD: float = 100.0  # microV - Flag channels exceeding this
GRADIENT_THRESHOLD: float = 50.0  # microV/sample - Detect rapid jumps
FLATLINE_THRESHOLD: float = 1e-6  # microV - Detect dead channels

# =============================================================================
# FREQUENCY BAND DEFINITIONS (Standard EEG/IFCN nomenclature)
# =============================================================================

FREQUENCY_BANDS: Dict[str, Tuple[float, float]] = {
    'delta': (0.5, 4.0),    # Deep sleep, unconscious processes
    'theta': (4.0, 8.0),    # Drowsiness, meditative states, memory encoding
    'alpha': (8.0, 13.0),   # Relaxed wakefulness, closed eyes
    'beta': (13.0, 30.0),   # Active thinking, focus, anxiety
    'gamma': (30.0, 45.0),  # High-level cognitive processing
}

# Sub-band divisions for enhanced feature extraction (Zheng & Lu, 2015)
ALPHA_SUB_BANDS: Dict[str, Tuple[float, float]] = {
    'alpha_low': (8.0, 10.0),
    'alpha_high': (10.0, 13.0),
}

BETA_SUB_BANDS: Dict[str, Tuple[float, float]] = {
    'beta_low': (13.0, 20.0),
    'beta_high': (20.0, 30.0),
}

# =============================================================================
# FRONTAL ALPHA ASYMMETRY (FAA) CONFIGURATION
# =============================================================================

# Asymmetry channel pairs (left-right)
# Based on Davidson (1992) and Frantzidis et al. (2010)
FAA_PAIRS: List[Tuple[str, str]] = [
    ('Fp1', 'Fp2'),  # Frontal pole asymmetry (primary)
    ('F3', 'F4'),    # Dorsolateral prefrontal cortex
    ('F7', 'F8'),    # Frontotemporal regions
]

# FAA computation method
FAA_METHOD: str = 'log_power'  # Options: 'log_power', 'raw_power', 'normalized'

# =============================================================================
# DEEP LEARNING MODEL HYPERPARAMETERS
# =============================================================================

# Model architecture (based on EEGNet and LSTM-RNN for emotion recognition)
MODEL_NAME: str = "CNN_BiLSTM_Hierarchical"

# Input shape parameters
N_CHANNELS: int = 32  # Default for full EEG setup (use 2 for minimal: Fp1, Fp2)
N_TIMEPOINTS: int = int(WINDOW_SIZE * SAMPLING_RATE)  # 512 samples
# N_FEATURES is dynamically calculated based on actual feature extraction:
# - Band power: 5 bands × N_CHANNELS = 5 × 32 = 160 features
# - FAA: len(FAA_PAIRS) = 3 features (if channel_names provided)
# - Statistics (optional): 6 × N_CHANNELS = 192 features
# - Spectral (optional): N_CHANNELS = 32 features
# Default (band power + FAA only): 160 + 3 = 163 features
# With stats: 160 + 3 + 192 = 355 features
N_FEATURES: int = 163  # Default: 5 bands × 32 channels + 3 FAA pairs

# CNN parameters
CNN_FILTERS: List[int] = [64, 128, 256]  # Progressive feature extraction
CNN_KERNEL_SIZE: int = 3
CNN_POOL_SIZE: int = 2
CNN_DROPOUT: float = 0.5

# BiLSTM parameters
LSTM_UNITS: int = 128  # Increased from 64 for better temporal modeling
LSTM_DROPOUT: float = 0.4
LSTM_RECURRENT_DROPOUT: float = 0.3

# Dense layer parameters
DENSE_UNITS: List[int] = [256, 128]  # Two-layer classifier
DENSE_DROPOUT: float = 0.5

# Hierarchical classification setup
VALENCE_CLASSES: int = 2  # Positive (1) vs Negative (0)
AROUSAL_CLASSES: int = 2  # High (1) vs Low (0)
EMOTION_CLASSES: int = 5  # happy, sad, relaxed, focused, neutral

# Emotion label mapping (extensible design)
EMOTION_LABELS: Dict[int, str] = {
    0: 'neutral',
    1: 'happy',
    2: 'sad',
    3: 'relaxed',
    4: 'focused',
}

# Reverse mapping for encoding
EMOTION_TO_ID: Dict[str, int] = {v: k for k, v in EMOTION_LABELS.items()}

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================

# Optimizer configuration
LEARNING_RATE: float = 0.001  # Adam default
OPTIMIZER: str = 'adam'  # Options: 'adam', 'sgd', 'rmsprop'

# Training parameters
BATCH_SIZE: int = 32  # Standard batch size for EEG (balanced speed/memory)
EPOCHS: int = 100  # With early stopping
VALIDATION_SPLIT: float = 0.2

# Early stopping configuration
EARLY_STOPPING_PATIENCE: int = 15
EARLY_STOPPING_MIN_DELTA: float = 0.001
EARLY_STOPPING_MONITOR: str = 'val_loss'

# Learning rate reduction
REDUCE_LR_PATIENCE: int = 7
REDUCE_LR_FACTOR: float = 0.5
REDUCE_LR_MIN_LR: float = 1e-6

# Model checkpointing
CHECKPOINT_MONITOR: str = 'val_accuracy'
CHECKPOINT_MODE: str = 'max'
CHECKPOINT_SAVE_BEST_ONLY: bool = True

# =============================================================================
# TRANSFER LEARNING / PERSONALIZATION PARAMETERS
# =============================================================================

# Fine-tuning configuration
FREEZE_LAYERS: int = 10  # Number of initial layers to freeze
FINETUNE_LEARNING_RATE: float = 0.0001  # Lower LR for fine-tuning
FINETUNE_EPOCHS: int = 20
FINETUNE_BATCH_SIZE: int = 8  # Smaller batches for limited personal data

# Minimum samples required for personalization
MIN_PERSONALIZATION_SAMPLES: int = 50  # Per emotion class

# =============================================================================
# DATA LOADING PARAMETERS
# =============================================================================

# Dataset formats supported
SUPPORTED_FORMATS: List[str] = ['.edf', '.mat', '.csv', '.pkl']

# DEAP dataset configuration
DEAP_SAMPLING_RATE: int = 128  # Hz
DEAP_CHANNELS: int = 32
DEAP_BASELINE_DURATION: float = 3.0  # seconds
DEAP_TRIAL_DURATION: float = 60.0  # seconds

# SEED dataset configuration
SEED_SAMPLING_RATE: int = 200  # Hz
SEED_CHANNELS: int = 62

# Data augmentation parameters
AUGMENTATION_ENABLED: bool = True
NOISE_LEVEL: float = 0.05  # 5% Gaussian noise for augmentation
TIME_SHIFT_MAX: int = 50  # Maximum samples to shift

# =============================================================================
# LIVE STREAMING PARAMETERS
# =============================================================================

# Serial/Bluetooth configuration
SERIAL_BAUD_RATE: int = 115200
SERIAL_TIMEOUT: float = 1.0  # seconds
BUFFER_SIZE: int = 2048  # samples
MAX_DROPPED_PACKETS: int = 10  # Before warning

# Real-time processing
PROCESSING_INTERVAL: float = 0.5  # seconds - Update rate
PREDICTION_SMOOTHING: int = 3  # Average over N predictions

# =============================================================================
# MUSIC RECOMMENDATION PARAMETERS
# =============================================================================

# Spotify API configuration (if using)
SPOTIFY_CLIENT_ID: str = os.getenv('SPOTIFY_CLIENT_ID', '')
SPOTIFY_CLIENT_SECRET: str = os.getenv('SPOTIFY_CLIENT_SECRET', '')

# Mood-to-music genre mapping
MOOD_GENRE_MAP: Dict[str, List[str]] = {
    'happy': ['pop', 'dance', 'funk', 'disco'],
    'sad': ['blues', 'indie', 'acoustic', 'classical'],
    'relaxed': ['ambient', 'chillout', 'jazz', 'lo-fi'],
    'focused': ['electronic', 'instrumental', 'classical', 'minimal'],
    'neutral': ['pop', 'rock', 'alternative', 'indie'],
}

# Recommendation parameters
SONGS_PER_RECOMMENDATION: int = 3
RECOMMENDATION_DIVERSITY: float = 0.7  # 0=similar, 1=diverse

# =============================================================================
# LOGGING AND DEBUGGING
# =============================================================================

# Logging configuration
LOG_LEVEL: str = 'INFO'  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_TO_FILE: bool = True
LOG_TO_CONSOLE: bool = True

# Debug flags
DEBUG_PLOT_SIGNALS: bool = False  # Plot preprocessed signals
DEBUG_SAVE_FEATURES: bool = False  # Save extracted features to CSV
DEBUG_VERBOSE: bool = False  # Extra print statements

# Performance monitoring
PROFILE_PERFORMANCE: bool = False  # Enable cProfile
TRACK_MEMORY: bool = False  # Track memory usage

# =============================================================================
# VALIDATION AND SANITY CHECKS
# =============================================================================

def calculate_n_features(
    n_channels: int = N_CHANNELS,
    n_bands: int = 5,
    n_faa_pairs: int = 3,
    include_statistics: bool = False,
    include_spectral: bool = False
) -> int:
    """
    Calculate the number of features based on extraction parameters.

    Args:
        n_channels: Number of EEG channels
        n_bands: Number of frequency bands (default: 5 for delta, theta, alpha, beta, gamma)
        n_faa_pairs: Number of FAA channel pairs (default: 3)
        include_statistics: Whether to include statistical features (mean, std, skew, kurtosis, ptp, rms)
        include_spectral: Whether to include spectral entropy features

    Returns:
        int: Total number of features

    Example:
        >>> # For 32 channels with band power and FAA only
        >>> n_features = calculate_n_features(32, include_statistics=False)
        >>> print(n_features)  # 163 (160 band powers + 3 FAA)

        >>> # For 32 channels with all features
        >>> n_features = calculate_n_features(32, include_statistics=True, include_spectral=True)
        >>> print(n_features)  # 387 (160 + 3 + 192 + 32)
    """
    # Band power features: n_bands × n_channels
    band_power_features = n_bands * n_channels

    # FAA features: n_faa_pairs (only if channel names support FAA)
    faa_features = n_faa_pairs

    # Statistical features: 6 statistics × n_channels
    # (mean, std, skewness, kurtosis, ptp, rms)
    statistical_features = 6 * n_channels if include_statistics else 0

    # Spectral features: n_channels (entropy for each channel)
    spectral_features = n_channels if include_spectral else 0

    total = band_power_features + faa_features + statistical_features + spectral_features

    return total


def validate_config() -> bool:
    """
    Validate configuration parameters for consistency.

    Returns:
        bool: True if configuration is valid, raises ValueError otherwise.
    """
    # Check sampling rate vs bandpass high
    if BANDPASS_HIGH >= SAMPLING_RATE / 2:
        raise ValueError(f"BANDPASS_HIGH ({BANDPASS_HIGH}) must be < Nyquist frequency ({SAMPLING_RATE/2})")

    # Check window size
    if WINDOW_SIZE <= 0:
        raise ValueError(f"WINDOW_SIZE must be positive, got {WINDOW_SIZE}")

    # Check overlap
    if not 0 <= OVERLAP < 1:
        raise ValueError(f"OVERLAP must be in [0, 1), got {OVERLAP}")

    # Check frequency bands
    for band_name, (low, high) in FREQUENCY_BANDS.items():
        if low >= high:
            raise ValueError(f"Invalid band {band_name}: low={low} >= high={high}")
        if high > BANDPASS_HIGH:
            raise ValueError(f"Band {band_name} high ({high}) exceeds BANDPASS_HIGH ({BANDPASS_HIGH})")

    # Check emotion classes
    if len(EMOTION_LABELS) != EMOTION_CLASSES:
        raise ValueError(f"EMOTION_LABELS length ({len(EMOTION_LABELS)}) != EMOTION_CLASSES ({EMOTION_CLASSES})")

    return True


# Run validation on import
validate_config()


# =============================================================================
# CONFIG CLASS WRAPPER (for compatibility with new modules)
# =============================================================================

class Config:
    """
    Configuration class wrapper for compatibility with new modules.
    
    Provides access to all configuration constants as class attributes.
    """
    
    # Paths
    BASE_DIR = BASE_DIR
    DATA_DIR = DATA_DIR
    MODEL_DIR = MODEL_DIR
    LOG_DIR = LOG_DIR
    MUSIC_DIR = MUSIC_DIR
    
    # EEG Parameters
    SAMPLING_RATE = SAMPLING_RATE
    WINDOW_SIZE = WINDOW_SIZE
    OVERLAP = OVERLAP
    N_CHANNELS = 32
    CHANNEL_NAMES = STANDARD_CHANNEL_NAMES[:32]  # First 32 channels
    
    # Preprocessing
    BANDPASS_LOWCUT = BANDPASS_LOW
    BANDPASS_HIGHCUT = BANDPASS_HIGH
    BANDPASS_ORDER = FILTER_ORDER  # Fixed: was BUTTER_ORDER
    NOTCH_FREQ = 50.0
    
    # Features
    FREQ_BANDS = FREQUENCY_BANDS  # Fixed: was FREQ_BANDS
    FAA_CHANNEL_PAIRS = [('F3', 'F4'), ('F7', 'F8'), ('Fp1', 'Fp2')]
    
    # Model
    N_EMOTION_CLASSES = EMOTION_CLASSES
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    
    def validate(self):
        """Validate configuration."""
        return validate_config()
    
    def __repr__(self):
        return f"Config(sampling_rate={self.SAMPLING_RATE}, n_channels={self.N_CHANNELS})"
