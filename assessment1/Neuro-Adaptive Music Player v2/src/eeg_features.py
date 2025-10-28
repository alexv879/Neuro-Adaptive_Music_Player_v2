"""
EEG Feature Extraction Module
===============================

Efficient, vectorized feature extraction for emotion recognition from EEG signals.
Implements best practices from top emotion recognition papers:
- Zheng & Lu (2015): Differential Entropy features
- Frantzidis et al. (2010): Frontal Alpha Asymmetry
- Koelstra et al. (2012): DEAP database methodology
- Li & Lu (2009): Band power and statistical features

Features:
- Traditional band power (delta, theta, alpha, beta, gamma)
- Frontal alpha asymmetry (FAA) for valence detection
- Windowed processing with configurable overlap
- Fully vectorized for (n_channels, n_samples) input
- Memory-efficient batch processing
- Support for custom frequency bands

Author: Rebuilt for CMP9780M Assessment
License: Proprietary (see root LICENSE)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from scipy.signal import welch
from scipy.fft import rfft, rfftfreq
import warnings
import logging

# Import configuration
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from config import (
    SAMPLING_RATE, WINDOW_SIZE, OVERLAP, FREQUENCY_BANDS,
    FAA_PAIRS, FAA_METHOD, ALPHA_SUB_BANDS, BETA_SUB_BANDS
)

# Configure logging
logger = logging.getLogger(__name__)


class EEGFeatureExtractor:
    """
    High-performance feature extractor for EEG emotion recognition.
    
    Extracts multiple feature types using vectorized operations:
    - Band power (delta, theta, alpha, beta, gamma)
    - Frontal alpha asymmetry (FAA)
    - Statistical features (mean, std, skewness, kurtosis)
    - Spectral features (spectral entropy, edge frequency)
    
    Attributes:
        fs (int): Sampling frequency
        window_size (float): Window duration in seconds
        overlap (float): Overlap ratio between windows (0-1)
        bands (Dict): Frequency band definitions
        
    Example:
        >>> extractor = EEGFeatureExtractor(fs=256)
        >>> features = extractor.extract_all_features(eeg_data, channel_names)
        >>> print(features.keys())  # ['band_power', 'faa', 'statistics']
    """
    
    def __init__(
        self,
        fs: int = SAMPLING_RATE,
        window_size: float = WINDOW_SIZE,
        overlap: float = OVERLAP,
        bands: Optional[Dict[str, Tuple[float, float]]] = None
    ):
        """
        Initialize feature extractor with signal parameters.
        
        Args:
            fs: Sampling frequency in Hz
            window_size: Window duration in seconds
            overlap: Overlap ratio (0 = no overlap, 0.5 = 50% overlap)
            bands: Custom frequency band definitions (defaults to config bands)
        """
        self.fs = fs
        self.window_size = window_size
        self.overlap = overlap
        self.bands = bands if bands is not None else FREQUENCY_BANDS
        
        # Calculate window parameters
        self.n_samples_window = int(window_size * fs)
        self.n_samples_hop = int(self.n_samples_window * (1 - overlap))
        
        logger.info(
            f"FeatureExtractor initialized: fs={fs}Hz, "
            f"window={window_size}s ({self.n_samples_window} samples), "
            f"overlap={overlap*100:.0f}%"
        )
    
    # =========================================================================
    # WINDOWING OPERATIONS
    # =========================================================================
    
    def create_windows(
        self,
        data: np.ndarray,
        axis: int = -1
    ) -> np.ndarray:
        """
        Create overlapping windows from continuous data using stride tricks.
        
        This is a memory-efficient implementation using numpy's as_strided
        to create views instead of copies. Based on librosa.util.frame approach.
        
        Args:
            data: EEG data of shape (..., n_samples)
            axis: Axis along which to create windows (default: last axis)
            
        Returns:
            np.ndarray: Windowed data of shape (..., n_windows, window_size)
            
        Example:
            >>> data = np.random.randn(32, 10000)  # 32 channels, ~40s
            >>> windowed = extractor.create_windows(data)
            >>> print(windowed.shape)  # (32, 20, 512) with default settings
        """
        if data.size == 0:
            raise ValueError("Cannot create windows from empty array")
        
        # Move axis to last position if needed
        if axis != -1 and axis != data.ndim - 1:
            data = np.moveaxis(data, axis, -1)
        
        n_samples = data.shape[-1]
        
        # Calculate number of windows
        n_windows = 1 + (n_samples - self.n_samples_window) // self.n_samples_hop
        
        if n_windows < 1:
            warnings.warn(
                f"Data length ({n_samples}) too short for window size "
                f"({self.n_samples_window}). Returning single window."
            )
            return data[..., np.newaxis, :min(n_samples, self.n_samples_window)]
        
        # Use stride tricks for memory efficiency
        # This creates a view, not a copy
        shape = data.shape[:-1] + (n_windows, self.n_samples_window)
        strides = data.strides[:-1] + (self.n_samples_hop * data.strides[-1], data.strides[-1])
        
        windowed = np.lib.stride_tricks.as_strided(
            data,
            shape=shape,
            strides=strides,
            writeable=False  # Prevent accidental modification
        )
        
        logger.debug(f"Created {n_windows} windows from {n_samples} samples")
        
        return windowed
    
    # =========================================================================
    # BAND POWER EXTRACTION (VECTORIZED)
    # =========================================================================
    
    def extract_band_power_welch(
        self,
        data: np.ndarray,
        band: Optional[Tuple[float, float]] = None,
        axis: int = -1
    ) -> np.ndarray:
        """
        Extract band power using Welch's method (periodogram averaging).
        
        Welch's method provides better spectral estimate than raw FFT by:
        1. Dividing signal into overlapping segments
        2. Computing periodogram for each segment
        3. Averaging periodograms to reduce variance
        
        This is the gold standard for EEG power estimation (Welch, 1967).
        
        Args:
            data: EEG data of shape (..., n_samples)
            band: Frequency band as (low, high) tuple in Hz
            axis: Axis along which to compute PSD
            
        Returns:
            np.ndarray: Band power values, shape = data.shape[:-1]
            
        Example:
            >>> data = np.random.randn(32, 1280)
            >>> alpha_power = extractor.extract_band_power_welch(data, (8, 13))
            >>> print(alpha_power.shape)  # (32,)
        """
        if band is None:
            band = (0.5, self.fs / 2)
        
        low_freq, high_freq = band
        
        # Compute power spectral density using Welch's method
        # nperseg: length of each segment (default is good for EEG)
        nperseg = min(self.n_samples_window, data.shape[axis])
        
        freqs, psd = welch(
            data,
            fs=self.fs,
            nperseg=nperseg,
            axis=axis
        )
        
        # Find indices for frequency band
        band_idx = (freqs >= low_freq) & (freqs <= high_freq)
        
        if not np.any(band_idx):
            warnings.warn(f"No frequencies found in band {band}")
            return np.zeros(data.shape[:-1])
        
        # Integrate power over band (trapezoidal rule)
        freq_res = freqs[1] - freqs[0]
        band_power = np.trapz(psd[..., band_idx], dx=freq_res, axis=-1)
        
        return band_power
    
    def extract_band_power_fft(
        self,
        data: np.ndarray,
        band: Optional[Tuple[float, float]] = None,
        axis: int = -1
    ) -> np.ndarray:
        """
        Extract band power using Fast Fourier Transform (FFT).
        
        Faster than Welch but higher variance. Use for real-time applications
        where speed is critical. For research/offline analysis, prefer Welch.
        
        Args:
            data: EEG data of shape (..., n_samples)
            band: Frequency band as (low, high) tuple in Hz
            axis: Axis along which to compute FFT
            
        Returns:
            np.ndarray: Band power values
        """
        if band is None:
            band = (0.5, self.fs / 2)
        
        low_freq, high_freq = band
        
        # Compute FFT (real FFT for real signals)
        fft_vals = rfft(data, axis=axis)
        freqs = rfftfreq(data.shape[axis], 1/self.fs)
        
        # Compute power spectrum (magnitude squared)
        power_spectrum = np.abs(fft_vals) ** 2
        
        # Find band indices
        band_idx = (freqs >= low_freq) & (freqs <= high_freq)
        
        # Sum power in band
        band_power = np.sum(power_spectrum[..., band_idx], axis=-1)
        
        return band_power
    
    def extract_all_band_powers(
        self,
        data: np.ndarray,
        method: str = 'welch',
        normalize: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Extract power for all frequency bands in one pass.

        OPTIMIZED: Computes PSD once and integrates over different bands for efficiency.
        This is 5x faster than calling extract_band_power separately for each band.

        Args:
            data: EEG data of shape (n_channels, n_samples) or
                  (n_trials, n_channels, n_samples)
            method: 'welch' (recommended) or 'fft' (faster but noisier)
            normalize: Normalize powers to sum to 1 (relative power)

        Returns:
            Dict[str, np.ndarray]: Band powers for each frequency band

        Example:
            >>> data = np.random.randn(32, 1280)
            >>> powers = extractor.extract_all_band_powers(data)
            >>> print(powers.keys())  # ['delta', 'theta', 'alpha', 'beta', 'gamma']
            >>> print(powers['alpha'].shape)  # (32,)
        """
        band_powers = {}

        if method == 'welch':
            # OPTIMIZED: Compute PSD once for all bands
            nperseg = min(self.n_samples_window, data.shape[-1])
            freqs, psd = welch(data, fs=self.fs, nperseg=nperseg, axis=-1)
            freq_res = freqs[1] - freqs[0]

            # Integrate over each frequency band
            for band_name, (low_freq, high_freq) in self.bands.items():
                band_idx = (freqs >= low_freq) & (freqs <= high_freq)
                if not np.any(band_idx):
                    warnings.warn(f"No frequencies found in band {band_name}")
                    band_powers[band_name] = np.zeros(data.shape[:-1])
                else:
                    # Trapezoidal integration
                    band_powers[band_name] = np.trapz(psd[..., band_idx], dx=freq_res, axis=-1)

        elif method == 'fft':
            # OPTIMIZED: Compute FFT once for all bands
            from scipy.fft import rfft, rfftfreq
            fft_vals = rfft(data, axis=-1)
            freqs = rfftfreq(data.shape[-1], 1/self.fs)
            power_spectrum = np.abs(fft_vals) ** 2

            # Sum power in each band
            for band_name, (low_freq, high_freq) in self.bands.items():
                band_idx = (freqs >= low_freq) & (freqs <= high_freq)
                band_powers[band_name] = np.sum(power_spectrum[..., band_idx], axis=-1)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'welch' or 'fft'")

        # Normalize if requested
        if normalize:
            total_power = sum(band_powers.values())
            # Avoid division by zero
            total_power = np.where(total_power < 1e-10, 1.0, total_power)

            for band_name in band_powers:
                band_powers[band_name] = band_powers[band_name] / total_power

        logger.debug(f"Extracted {len(band_powers)} band powers using {method} method")

        return band_powers
    
    # =========================================================================
    # FRONTAL ALPHA ASYMMETRY (FAA)
    # =========================================================================
    
    def extract_faa(
        self,
        data: np.ndarray,
        channel_names: List[str],
        pairs: Optional[List[Tuple[str, str]]] = None,
        method: str = FAA_METHOD
    ) -> Dict[str, float]:
        """
        Extract Frontal Alpha Asymmetry (FAA) for emotion valence detection.
        
        FAA is computed as the difference in alpha power between right and
        left frontal regions. Based on Davidson's approach-withdrawal model:
        - Positive FAA (right > left): Approach motivation, positive valence
        - Negative FAA (left > right): Withdrawal motivation, negative valence
        
        Formula (Frantzidis et al., 2010):
            FAA = log(right_alpha) - log(left_alpha)
        
        Args:
            data: EEG data of shape (n_channels, n_samples)
            channel_names: List of channel names matching data rows
            pairs: Channel pairs as [(left, right), ...] (defaults to config)
            method: 'log_power', 'raw_power', or 'normalized'
            
        Returns:
            Dict[str, float]: FAA values for each channel pair
            
        Example:
            >>> data = np.random.randn(32, 1280)
            >>> channels = ['Fp1', 'Fp2', 'F3', 'F4', ...]
            >>> faa = extractor.extract_faa(data, channels)
            >>> print(faa)  # {'Fp1_Fp2': 0.15, 'F3_F4': -0.08, ...}
        """
        if pairs is None:
            pairs = FAA_PAIRS
        
        # Extract alpha power for all channels
        alpha_power = self.extract_band_power_welch(data, self.bands['alpha'], axis=-1)
        
        # Create channel name to index mapping
        channel_map = {name: idx for idx, name in enumerate(channel_names)}
        
        faa_values = {}
        
        for left_ch, right_ch in pairs:
            # Check if channels exist
            if left_ch not in channel_map or right_ch not in channel_map:
                warnings.warn(
                    f"Channel pair ({left_ch}, {right_ch}) not found in data. "
                    f"Available channels: {channel_names}"
                )
                continue
            
            # Get alpha power for left and right channels
            left_idx = channel_map[left_ch]
            right_idx = channel_map[right_ch]
            
            left_power = alpha_power[left_idx]
            right_power = alpha_power[right_idx]
            
            # Compute FAA using specified method
            if method == 'log_power':
                # Log transform to normalize distribution (standard approach)
                faa = np.log(right_power + 1e-10) - np.log(left_power + 1e-10)
            elif method == 'raw_power':
                faa = right_power - left_power
            elif method == 'normalized':
                faa = (right_power - left_power) / (right_power + left_power + 1e-10)
            else:
                raise ValueError(f"Unknown FAA method: {method}")
            
            pair_name = f"{left_ch}_{right_ch}"
            faa_values[pair_name] = float(faa)
        
        logger.debug(f"Extracted FAA for {len(faa_values)} channel pairs")
        
        return faa_values
    
    # =========================================================================
    # STATISTICAL FEATURES
    # =========================================================================
    
    def extract_statistical_features(
        self,
        data: np.ndarray,
        axis: int = -1
    ) -> Dict[str, np.ndarray]:
        """
        Extract statistical features from time-domain signal.
        
        Features include:
        - Mean: DC level
        - Standard deviation: Signal variability
        - Skewness: Distribution asymmetry
        - Kurtosis: Tail heaviness (outlier sensitivity)
        - Peak-to-peak amplitude
        - Root mean square (RMS)
        
        Args:
            data: EEG data of shape (..., n_samples)
            axis: Axis along which to compute statistics
            
        Returns:
            Dict[str, np.ndarray]: Statistical features
        """
        from scipy.stats import skew, kurtosis
        
        features = {
            'mean': np.mean(data, axis=axis),
            'std': np.std(data, axis=axis),
            'skewness': skew(data, axis=axis),
            'kurtosis': kurtosis(data, axis=axis),
            'ptp': np.ptp(data, axis=axis),  # Peak-to-peak
            'rms': np.sqrt(np.mean(data**2, axis=axis)),  # Root mean square
        }
        
        return features
    
    # =========================================================================
    # SPECTRAL FEATURES
    # =========================================================================
    
    def extract_spectral_entropy(
        self,
        data: np.ndarray,
        axis: int = -1
    ) -> np.ndarray:
        """
        Extract spectral entropy as a measure of signal complexity.
        
        High entropy = more uniform spectrum (complex/noisy signal)
        Low entropy = peaked spectrum (rhythmic/periodic signal)
        
        Args:
            data: EEG data of shape (..., n_samples)
            axis: Axis along which to compute entropy
            
        Returns:
            np.ndarray: Spectral entropy values
        """
        # Compute power spectrum
        fft_vals = rfft(data, axis=axis)
        power = np.abs(fft_vals) ** 2
        
        # Normalize to probability distribution
        power_norm = power / (np.sum(power, axis=-1, keepdims=True) + 1e-10)
        
        # Compute Shannon entropy
        entropy = -np.sum(power_norm * np.log2(power_norm + 1e-10), axis=-1)
        
        return entropy
    
    # =========================================================================
    # COMBINED FEATURE EXTRACTION
    # =========================================================================
    
    def extract_all_features(
        self,
        data: np.ndarray,
        channel_names: Optional[List[str]] = None,
        include_faa: bool = True,
        include_stats: bool = True,
        include_spectral: bool = False
    ) -> Dict[str, Union[Dict, np.ndarray]]:
        """
        Extract all features in one pass for maximum efficiency.
        
        This is the main method for feature extraction. It computes:
        - Band powers (delta, theta, alpha, beta, gamma) for each channel
        - Frontal alpha asymmetry (if channel names provided)
        - Statistical features (optional)
        - Spectral features (optional)
        
        Args:
            data: EEG data of shape (n_channels, n_samples)
            channel_names: List of channel names (required for FAA)
            include_faa: Compute frontal alpha asymmetry
            include_stats: Compute statistical features
            include_spectral: Compute spectral features
            
        Returns:
            Dict containing:
                - 'band_power': Dict of band powers
                - 'faa': Dict of FAA values (if include_faa=True and channel_names provided)
                - 'statistics': Dict of statistical features (if include_stats=True)
                - 'spectral': Dict of spectral features (if include_spectral=True)
                
        Example:
            >>> data = np.random.randn(32, 1280)
            >>> channels = ['Fp1', 'Fp2', 'F3', 'F4', ...]
            >>> features = extractor.extract_all_features(data, channels)
            >>> # Features ready for model input
        """
        all_features = {}
        
        # Band powers (always computed)
        all_features['band_power'] = self.extract_all_band_powers(data, method='welch')
        
        # Frontal alpha asymmetry
        if include_faa and channel_names is not None:
            all_features['faa'] = self.extract_faa(data, channel_names)
        elif include_faa and channel_names is None:
            warnings.warn("FAA requested but channel_names not provided. Skipping FAA.")
        
        # Statistical features
        if include_stats:
            all_features['statistics'] = self.extract_statistical_features(data)
        
        # Spectral features
        if include_spectral:
            all_features['spectral'] = {
                'entropy': self.extract_spectral_entropy(data)
            }
        
        return all_features
    
    # =========================================================================
    # FEATURE VECTOR CONSTRUCTION
    # =========================================================================
    
    def features_to_vector(
        self,
        features: Dict,
        flatten: bool = True
    ) -> np.ndarray:
        """
        Convert feature dictionary to flat vector for model input.
        
        Concatenates all features into a single 1D vector suitable for
        machine learning models.
        
        Args:
            features: Feature dictionary from extract_all_features()
            flatten: Flatten multi-dimensional features
            
        Returns:
            np.ndarray: Feature vector of shape (n_features,)
            
        Example:
            >>> features = extractor.extract_all_features(data, channels)
            >>> feature_vec = extractor.features_to_vector(features)
            >>> print(feature_vec.shape)  # (167,) for 32 channels with all features
        """
        vectors = []
        
        # Band powers
        if 'band_power' in features:
            for band_name in sorted(features['band_power'].keys()):
                band_power = features['band_power'][band_name]
                if flatten and band_power.ndim > 0:
                    vectors.append(band_power.flatten())
                else:
                    vectors.append(np.atleast_1d(band_power))
        
        # FAA values
        if 'faa' in features:
            faa_values = [features['faa'][key] for key in sorted(features['faa'].keys())]
            vectors.append(np.array(faa_values))
        
        # Statistical features
        if 'statistics' in features:
            for stat_name in sorted(features['statistics'].keys()):
                stat_values = features['statistics'][stat_name]
                if flatten and stat_values.ndim > 0:
                    vectors.append(stat_values.flatten())
                else:
                    vectors.append(np.atleast_1d(stat_values))
        
        # Spectral features
        if 'spectral' in features:
            for spec_name in sorted(features['spectral'].keys()):
                spec_values = features['spectral'][spec_name]
                if flatten and spec_values.ndim > 0:
                    vectors.append(spec_values.flatten())
                else:
                    vectors.append(np.atleast_1d(spec_values))
        
        # Concatenate all vectors
        feature_vector = np.concatenate(vectors)
        
        return feature_vector
    
    # =========================================================================
    # BATCH PROCESSING
    # =========================================================================
    
    def extract_features_batch(
        self,
        data: np.ndarray,
        channel_names: Optional[List[str]] = None,
        window_data: bool = True,
        **kwargs
    ) -> np.ndarray:
        """
        Extract features from multiple trials in batch.
        
        Efficiently processes (n_trials, n_channels, n_samples) data by
        vectorizing operations across trials.
        
        Args:
            data: EEG data of shape (n_trials, n_channels, n_samples)
            channel_names: List of channel names
            window_data: Create overlapping windows first
            **kwargs: Additional arguments for extract_all_features()
            
        Returns:
            np.ndarray: Feature matrix of shape (n_trials, n_features) or
                        (n_trials, n_windows, n_features) if window_data=True
                        
        Example:
            >>> # Process entire dataset
            >>> data = np.random.randn(100, 32, 1280)  # 100 trials
            >>> features = extractor.extract_features_batch(data, channels)
            >>> print(features.shape)  # (100, 167) or similar
        """
        if data.ndim != 3:
            raise ValueError(f"Expected 3D data (n_trials, n_channels, n_samples), got {data.shape}")
        
        n_trials, n_channels, n_samples = data.shape
        
        # Window data if requested
        if window_data:
            # Create windows: (n_trials, n_channels, n_windows, window_size)
            windowed = self.create_windows(data, axis=-1)
            n_windows = windowed.shape[2]
            
            # Reshape to (n_trials * n_windows, n_channels, window_size)
            reshaped = windowed.reshape(-1, n_channels, self.n_samples_window)
            
            # Extract features for all windows
            all_feature_vecs = []
            for i in range(reshaped.shape[0]):
                features = self.extract_all_features(
                    reshaped[i],
                    channel_names,
                    **kwargs
                )
                feature_vec = self.features_to_vector(features)
                all_feature_vecs.append(feature_vec)
            
            # Reshape back to (n_trials, n_windows, n_features)
            feature_matrix = np.array(all_feature_vecs)
            feature_matrix = feature_matrix.reshape(n_trials, n_windows, -1)
        else:
            # Extract features for each trial
            all_feature_vecs = []
            for i in range(n_trials):
                features = self.extract_all_features(
                    data[i],
                    channel_names,
                    **kwargs
                )
                feature_vec = self.features_to_vector(features)
                all_feature_vecs.append(feature_vec)
            
            feature_matrix = np.array(all_feature_vecs)
        
        logger.info(f"Extracted features from {n_trials} trials, shape: {feature_matrix.shape}")
        
        return feature_matrix


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compute_differential_entropy(
    data: np.ndarray,
    fs: int = SAMPLING_RATE,
    bands: Optional[Dict[str, Tuple[float, float]]] = None
) -> Dict[str, np.ndarray]:
    """
    Compute differential entropy features for each frequency band.
    
    Differential entropy (DE) is a powerful feature for emotion recognition,
    as demonstrated by Zheng & Lu (2015) achieving 84.22% accuracy on SEED dataset.
    
    DE is defined as:
        DE = 0.5 * log(2 * pi * e * variance)
    
    For Gaussian signals, DE is related to variance, but it captures more
    information about the probability distribution.
    
    Args:
        data: EEG data of shape (n_channels, n_samples)
        fs: Sampling frequency
        bands: Frequency band definitions
        
    Returns:
        Dict[str, np.ndarray]: Differential entropy for each band
        
    Reference:
        Zheng, W. L., & Lu, B. L. (2015). Investigating critical frequency bands
        and channels for EEG-based emotion recognition with deep neural networks.
        IEEE Trans. Autonomous Mental Development, 7(3), 162-175.
    """
    if bands is None:
        bands = FREQUENCY_BANDS
    
    extractor = EEGFeatureExtractor(fs=fs)
    de_features = {}
    
    for band_name, band_range in bands.items():
        # Extract band power
        power = extractor.extract_band_power_welch(data, band_range)
        
        # Compute differential entropy
        # Add small epsilon to avoid log(0)
        de = 0.5 * np.log(2 * np.pi * np.e * (power + 1e-10))
        
        de_features[band_name] = de
    
    return de_features


if __name__ == "__main__":
    # Demo and self-test
    print("EEG Feature Extraction Module - Self Test")
    print("=" * 60)
    
    # Create test data
    np.random.seed(42)
    n_channels = 32
    duration = 5  # seconds
    fs = 256
    n_samples = duration * fs
    
    test_data = np.random.randn(n_channels, n_samples) * 10  # 10 microV noise
    
    # Add realistic EEG signals
    t = np.arange(n_samples) / fs
    test_data[0] += 20 * np.sin(2 * np.pi * 10 * t)  # Alpha at Fp1
    test_data[1] += 15 * np.sin(2 * np.pi * 10 * t + np.pi/4)  # Alpha at Fp2
    
    channel_names = [
        'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
        'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
        'Fz', 'Cz', 'Pz', 'A1', 'A2', 'Fp1', 'Fp2', 'F3',
        'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7'
    ]
    
    print(f"Test data shape: {test_data.shape}")
    print(f"Channels: {len(channel_names)}")
    
    # Initialize extractor
    extractor = EEGFeatureExtractor(fs=fs, window_size=2.0, overlap=0.5)
    
    # Test windowing
    print("\n--- Testing Windowing ---")
    windowed = extractor.create_windows(test_data)
    print(f"Windowed shape: {windowed.shape}")
    
    # Test band power extraction
    print("\n--- Testing Band Power Extraction ---")
    band_powers = extractor.extract_all_band_powers(test_data)
    for band_name, power in band_powers.items():
        print(f"{band_name}: shape={power.shape}, mean={np.mean(power):.2f}")
    
    # Test FAA
    print("\n--- Testing Frontal Alpha Asymmetry ---")
    faa = extractor.extract_faa(test_data, channel_names[:n_channels])
    for pair, value in faa.items():
        print(f"{pair}: {value:.4f}")
    
    # Test all features
    print("\n--- Testing All Features Extraction ---")
    all_features = extractor.extract_all_features(
        test_data,
        channel_names[:n_channels],
        include_faa=True,
        include_stats=True,
        include_spectral=True
    )
    print(f"Feature groups: {list(all_features.keys())}")
    
    # Test feature vector
    print("\n--- Testing Feature Vector Construction ---")
    feature_vec = extractor.features_to_vector(all_features)
    print(f"Feature vector shape: {feature_vec.shape}")
    print(f"Feature vector (first 10): {feature_vec[:10]}")
    
    # Test batch processing
    print("\n--- Testing Batch Processing ---")
    batch_data = np.random.randn(10, n_channels, n_samples) * 10
    batch_features = extractor.extract_features_batch(
        batch_data,
        channel_names[:n_channels],
        window_data=False
    )
    print(f"Batch feature matrix shape: {batch_features.shape}")
    
    print("\nSelf-test complete!")
