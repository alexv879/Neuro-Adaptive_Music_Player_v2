"""
EEG Signal Preprocessing Module
================================

Robust, vectorized preprocessing pipeline for EEG data supporting both
batch and streaming modes. Implements best practices from:
- MNE-Python (Gramfort et al., 2013)
- EEGLAB (Delorme & Makeig, 2004)
- BCI Competition preprocessing guidelines
- Clean Rawdata plugin (Kothe & Makeig, 2013)

Features:
- Vectorized operations for efficiency
- Bandpass and notch filtering (Butterworth IIR)
- Artifact detection with multiple methods
- Hooks for ICA and IMU-based correction (future)
- Robust to missing data and shape variations
- Memory-efficient batch processing

Author: Rebuilt for CMP9780M Assessment
License: Proprietary (see root LICENSE)
"""

import numpy as np
import warnings
from typing import Optional, Tuple, Union, List
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt, welch
from scipy.stats import zscore
import logging

# Import configuration
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from config import (
    SAMPLING_RATE, BANDPASS_LOW, BANDPASS_HIGH, FILTER_ORDER,
    NOTCH_FREQ, NOTCH_Q, VOLTAGE_THRESHOLD, GRADIENT_THRESHOLD,
    FLATLINE_THRESHOLD
)

# Configure logging
logger = logging.getLogger(__name__)


class EEGPreprocessor:
    """
    High-performance EEG preprocessing pipeline with artifact rejection.
    
    Supports both batch processing (n_trials, n_channels, n_samples) and
    streaming mode (n_channels, n_samples) with automatic shape handling.
    
    Attributes:
        fs (int): Sampling frequency in Hz
        bandpass_sos (ndarray): Second-order sections for bandpass filter
        notch_b (ndarray): Notch filter numerator coefficients
        notch_a (ndarray): Notch filter denominator coefficients
        
    Example:
        >>> preprocessor = EEGPreprocessor(fs=256)
        >>> clean_data = preprocessor.preprocess(raw_eeg, apply_notch=True)
        >>> artifact_mask = preprocessor.detect_artifacts(clean_data)
    """
    
    def __init__(
        self,
        fs: int = SAMPLING_RATE,
        bandpass_low: float = BANDPASS_LOW,
        bandpass_high: float = BANDPASS_HIGH,
        filter_order: int = FILTER_ORDER,
        notch_freq: float = NOTCH_FREQ,
        notch_q: float = NOTCH_Q
    ):
        """
        Initialize the EEG preprocessor with filter parameters.
        
        Args:
            fs: Sampling frequency in Hz
            bandpass_low: Lower cutoff frequency for bandpass filter (Hz)
            bandpass_high: Upper cutoff frequency for bandpass filter (Hz)
            filter_order: Order of Butterworth filter
            notch_freq: Powerline frequency to notch filter (50 or 60 Hz)
            notch_q: Quality factor of notch filter (higher = narrower)
            
        Raises:
            ValueError: If parameters violate Nyquist theorem or are invalid
        """
        self.fs = fs
        self.bandpass_low = bandpass_low
        self.bandpass_high = bandpass_high
        self.filter_order = filter_order
        self.notch_freq = notch_freq
        self.notch_q = notch_q
        
        # Validate parameters
        self._validate_parameters()
        
        # Design filters
        self.bandpass_sos = self._design_bandpass_filter()
        self.notch_b, self.notch_a = self._design_notch_filter()
        
        logger.info(f"EEGPreprocessor initialized: fs={fs}Hz, "
                   f"bandpass=[{bandpass_low}-{bandpass_high}]Hz")
    
    def _validate_parameters(self) -> None:
        """Validate preprocessing parameters."""
        nyquist = self.fs / 2
        
        if self.bandpass_high >= nyquist:
            raise ValueError(
                f"bandpass_high ({self.bandpass_high}Hz) must be < "
                f"Nyquist frequency ({nyquist}Hz)"
            )
        
        if self.bandpass_low <= 0:
            raise ValueError(f"bandpass_low must be positive, got {self.bandpass_low}")
        
        if self.bandpass_low >= self.bandpass_high:
            raise ValueError(
                f"bandpass_low ({self.bandpass_low}) must be < "
                f"bandpass_high ({self.bandpass_high})"
            )
        
        if self.filter_order < 1:
            raise ValueError(f"filter_order must be >= 1, got {self.filter_order}")
    
    def _design_bandpass_filter(self) -> np.ndarray:
        """
        Design Butterworth bandpass filter using second-order sections (SOS).
        
        SOS format is preferred over ba format for numerical stability,
        especially for high-order filters. See scipy.signal.butter documentation.
        
        Returns:
            np.ndarray: Second-order sections representation of filter
        """
        nyquist = self.fs / 2
        low_norm = self.bandpass_low / nyquist
        high_norm = self.bandpass_high / nyquist
        
        # Design using SOS for numerical stability
        sos = butter(
            self.filter_order,
            [low_norm, high_norm],
            btype='bandpass',
            output='sos'
        )
        
        logger.debug(f"Designed bandpass filter: {self.bandpass_low}-{self.bandpass_high}Hz, "
                    f"order={self.filter_order}")
        
        return sos
    
    def _design_notch_filter(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Design IIR notch filter for powerline noise removal.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (numerator, denominator) coefficients
        """
        w0 = self.notch_freq / (self.fs / 2)  # Normalized frequency
        b, a = iirnotch(w0, self.notch_q)
        
        logger.debug(f"Designed notch filter: {self.notch_freq}Hz, Q={self.notch_q}")
        
        return b, a
    
    def apply_bandpass(
        self,
        data: np.ndarray,
        axis: int = -1
    ) -> np.ndarray:
        """
        Apply zero-phase bandpass filter to EEG data.
        
        Uses sosfiltfilt for zero-phase filtering with second-order sections,
        which is more numerically stable than filtfilt with ba coefficients.
        
        Args:
            data: EEG data of shape (..., n_samples) or any shape where
                  filtering is applied along the specified axis
            axis: Axis along which to filter (default: last axis)
            
        Returns:
            np.ndarray: Filtered data with same shape as input
            
        Example:
            >>> raw = np.random.randn(32, 1280)  # 32 channels, 5 seconds at 256Hz
            >>> filtered = preprocessor.apply_bandpass(raw, axis=-1)
        """
        if data.size == 0:
            warnings.warn("Empty array passed to apply_bandpass, returning as-is")
            return data
        
        # Check minimum length for filter
        min_length = 3 * max(len(self.bandpass_sos), 1)
        if data.shape[axis] < min_length:
            warnings.warn(
                f"Data length ({data.shape[axis]}) < minimum for filter ({min_length}), "
                f"skipping bandpass filtering"
            )
            return data
        
        try:
            # Apply zero-phase filtering using SOS
            filtered = sosfiltfilt(self.bandpass_sos, data, axis=axis)
            return filtered
        except Exception as e:
            logger.error(f"Bandpass filtering failed: {e}")
            raise
    
    def apply_notch(
        self,
        data: np.ndarray,
        axis: int = -1
    ) -> np.ndarray:
        """
        Apply zero-phase notch filter to remove powerline noise.
        
        Args:
            data: EEG data of shape (..., n_samples)
            axis: Axis along which to filter (default: last axis)
            
        Returns:
            np.ndarray: Notch-filtered data with same shape as input
        """
        if data.size == 0:
            warnings.warn("Empty array passed to apply_notch, returning as-is")
            return data
        
        try:
            # Apply zero-phase notch filtering
            filtered = filtfilt(self.notch_b, self.notch_a, data, axis=axis)
            return filtered
        except Exception as e:
            logger.error(f"Notch filtering failed: {e}")
            raise
    
    def remove_dc_offset(
        self,
        data: np.ndarray,
        axis: int = -1
    ) -> np.ndarray:
        """
        Remove DC offset by subtracting mean along time axis.
        
        This is a simple and effective method for removing DC drift,
        commonly used before further filtering (MNE-Python approach).
        
        Args:
            data: EEG data of shape (..., n_samples)
            axis: Axis along which to compute mean (default: last axis)
            
        Returns:
            np.ndarray: Zero-mean data
        """
        return data - np.mean(data, axis=axis, keepdims=True)
    
    def standardize(
        self,
        data: np.ndarray,
        axis: int = -1
    ) -> np.ndarray:
        """
        Standardize data to zero mean and unit variance (z-score).
        
        Useful for normalizing across channels or trials before model input.
        
        Args:
            data: EEG data of shape (..., n_samples)
            axis: Axis along which to standardize (default: last axis)
            
        Returns:
            np.ndarray: Standardized data
        """
        mean = np.mean(data, axis=axis, keepdims=True)
        std = np.std(data, axis=axis, keepdims=True)
        
        # Avoid division by zero
        std = np.where(std < 1e-10, 1.0, std)
        
        return (data - mean) / std
    
    def detect_artifacts(
        self,
        data: np.ndarray,
        method: str = 'all'
    ) -> np.ndarray:
        """
        Detect artifacts in EEG data using multiple criteria.
        
        Implements three detection methods:
        1. Voltage threshold: Detects excessive amplitude
        2. Gradient threshold: Detects rapid jumps (muscle artifacts)
        3. Flatline detection: Detects dead channels
        
        Args:
            data: EEG data of shape (n_channels, n_samples) or
                  (n_trials, n_channels, n_samples)
            method: Detection method ('voltage', 'gradient', 'flatline', 'all')
            
        Returns:
            np.ndarray: Boolean mask of shape matching input, where True indicates
                        artifact-free samples
                        
        Example:
            >>> raw = np.random.randn(32, 1280)
            >>> mask = preprocessor.detect_artifacts(raw, method='all')
            >>> clean = raw * mask  # Zero out artifacts
        """
        original_shape = data.shape
        
        # Ensure 3D: (n_trials, n_channels, n_samples)
        if data.ndim == 2:
            data = data[np.newaxis, :, :]
        elif data.ndim != 3:
            raise ValueError(f"Data must be 2D or 3D, got shape {original_shape}")
        
        n_trials, n_channels, n_samples = data.shape
        artifact_mask = np.ones_like(data, dtype=bool)
        
        if method in ['voltage', 'all']:
            # Detect voltage threshold violations
            voltage_violations = np.abs(data) > VOLTAGE_THRESHOLD
            artifact_mask &= ~voltage_violations
            n_violations = np.sum(voltage_violations)
            if n_violations > 0:
                logger.warning(f"Voltage threshold violations: {n_violations} samples")
        
        if method in ['gradient', 'all']:
            # Detect rapid jumps (muscle artifacts)
            gradient = np.abs(np.diff(data, axis=-1))
            gradient_violations = gradient > GRADIENT_THRESHOLD
            # Pad to match original length
            gradient_violations = np.pad(
                gradient_violations,
                ((0, 0), (0, 0), (0, 1)),
                mode='edge'
            )
            artifact_mask &= ~gradient_violations
            n_violations = np.sum(gradient_violations)
            if n_violations > 0:
                logger.warning(f"Gradient threshold violations: {n_violations} samples")
        
        if method in ['flatline', 'all']:
            # Detect flatline channels (variance below threshold)
            variance = np.var(data, axis=-1, keepdims=True)
            flatline_channels = variance < FLATLINE_THRESHOLD
            # Broadcast to full shape
            flatline_mask = np.broadcast_to(flatline_channels, data.shape)
            artifact_mask &= ~flatline_mask
            n_flatline = np.sum(np.any(flatline_channels, axis=-1))
            if n_flatline > 0:
                logger.warning(f"Flatline channels detected: {n_flatline}")
        
        # Reshape back to original
        if len(original_shape) == 2:
            artifact_mask = artifact_mask[0]
        
        return artifact_mask
    
    def interpolate_bad_channels(
        self,
        data: np.ndarray,
        bad_mask: np.ndarray
    ) -> np.ndarray:
        """
        Interpolate bad channels using spherical spline interpolation.
        
        This is a placeholder for future implementation. In production,
        use MNE-Python's interpolate_bads() method or implement spherical
        spline interpolation following Perrin et al. (1989).
        
        Args:
            data: EEG data of shape (n_channels, n_samples)
            bad_mask: Boolean mask of shape (n_channels,) where True = bad
            
        Returns:
            np.ndarray: Data with interpolated channels
            
        Notes:
            Current implementation uses simple linear interpolation as placeholder.
            TODO: Implement proper spherical spline interpolation.
        """
        if not np.any(bad_mask):
            return data
        
        logger.warning(
            "Spherical spline interpolation not yet implemented. "
            "Using simple average interpolation as placeholder."
        )
        
        # Simple placeholder: replace bad channels with average of neighbors
        interpolated = data.copy()
        bad_indices = np.where(bad_mask)[0]
        good_indices = np.where(~bad_mask)[0]
        
        if len(good_indices) == 0:
            logger.error("All channels are bad, cannot interpolate")
            return data
        
        # Replace bad channels with mean of good channels
        interpolated[bad_indices] = np.mean(data[good_indices], axis=0, keepdims=True)
        
        return interpolated
    
    def preprocess(
        self,
        data: np.ndarray,
        apply_notch: bool = True,
        remove_dc: bool = True,
        standardize: bool = False,
        detect_artifacts: bool = False,
        interpolate_bad: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Complete preprocessing pipeline with configurable steps.
        
        Pipeline order:
        1. Remove DC offset (if enabled)
        2. Apply bandpass filter (always)
        3. Apply notch filter (if enabled)
        4. Detect artifacts (if enabled)
        5. Interpolate bad channels (if enabled)
        6. Standardize (if enabled)
        
        Args:
            data: EEG data of shape (n_channels, n_samples) or
                  (n_trials, n_channels, n_samples)
            apply_notch: Apply powerline notch filter
            remove_dc: Remove DC offset before filtering
            standardize: Standardize to zero mean and unit variance
            detect_artifacts: Return artifact mask
            interpolate_bad: Interpolate artifact-contaminated channels
            
        Returns:
            If detect_artifacts=False:
                np.ndarray: Preprocessed data
            If detect_artifacts=True:
                Tuple[np.ndarray, np.ndarray]: (preprocessed_data, artifact_mask)
                
        Example:
            >>> # Basic preprocessing
            >>> clean = preprocessor.preprocess(raw, apply_notch=True)
            >>> 
            >>> # With artifact detection
            >>> clean, mask = preprocessor.preprocess(
            ...     raw,
            ...     apply_notch=True,
            ...     detect_artifacts=True
            ... )
        """
        if data.size == 0:
            raise ValueError("Cannot preprocess empty array")
        
        logger.info(f"Preprocessing data of shape {data.shape}")
        
        # Store original shape
        original_shape = data.shape
        
        # Step 1: Remove DC offset
        if remove_dc:
            data = self.remove_dc_offset(data, axis=-1)
            logger.debug("Removed DC offset")
        
        # Step 2: Apply bandpass filter (always)
        data = self.apply_bandpass(data, axis=-1)
        logger.debug("Applied bandpass filter")
        
        # Step 3: Apply notch filter
        if apply_notch:
            data = self.apply_notch(data, axis=-1)
            logger.debug("Applied notch filter")
        
        # Step 4: Detect artifacts
        artifact_mask = None
        if detect_artifacts or interpolate_bad:
            artifact_mask = self.detect_artifacts(data, method='all')
            logger.debug(f"Detected artifacts: {np.sum(~artifact_mask)} samples")
        
        # Step 5: Interpolate bad channels
        if interpolate_bad and artifact_mask is not None:
            # Identify bad channels (>50% artifacts)
            if data.ndim == 2:
                bad_channel_mask = np.mean(~artifact_mask, axis=-1) > 0.5
            else:
                bad_channel_mask = np.mean(~artifact_mask, axis=(0, 2)) > 0.5
            
            if np.any(bad_channel_mask):
                if data.ndim == 2:
                    data = self.interpolate_bad_channels(data, bad_channel_mask)
                else:
                    # Process each trial separately
                    for i in range(data.shape[0]):
                        data[i] = self.interpolate_bad_channels(data[i], bad_channel_mask)
                logger.debug(f"Interpolated {np.sum(bad_channel_mask)} bad channels")
        
        # Step 6: Standardize
        if standardize:
            data = self.standardize(data, axis=-1)
            logger.debug("Standardized data")
        
        logger.info("Preprocessing complete")
        
        if detect_artifacts:
            return data, artifact_mask
        return data
    
    # =========================================================================
    # STREAMING MODE SUPPORT
    # =========================================================================
    
    def preprocess_stream(
        self,
        data: np.ndarray,
        state: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Preprocess streaming data with filter state preservation.
        
        For real-time applications, filter state must be preserved between
        consecutive chunks to avoid transients. This method uses lfilter
        instead of filtfilt for causal filtering.
        
        Args:
            data: EEG data chunk of shape (n_channels, n_samples)
            state: Filter state from previous call (None for first call)
            
        Returns:
            Tuple[np.ndarray, dict]: (preprocessed_data, new_state)
            
        Notes:
            This is a placeholder for future streaming implementation.
            TODO: Implement stateful filtering using scipy.signal.lfilter_zi
        """
        # Placeholder implementation - use regular preprocessing
        logger.warning(
            "Streaming mode not fully implemented. "
            "Using batch preprocessing as placeholder."
        )
        
        preprocessed = self.preprocess(
            data,
            apply_notch=True,
            remove_dc=True,
            standardize=False,
            detect_artifacts=False
        )
        
        # Return dummy state
        new_state = {'initialized': True}
        
        return preprocessed, new_state
    
    # =========================================================================
    # FUTURE: ICA AND IMU-BASED ARTIFACT CORRECTION HOOKS
    # =========================================================================
    
    def apply_ica_correction(
        self,
        data: np.ndarray,
        ica_weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Apply ICA-based artifact correction (placeholder).
        
        Future implementation will use FastICA or Infomax ICA for:
        - Eye blink removal
        - Muscle artifact reduction
        - Cardiac artifact suppression
        
        Args:
            data: EEG data
            ica_weights: Pre-computed ICA unmixing matrix
            
        Returns:
            np.ndarray: ICA-corrected data
            
        Notes:
            TODO: Implement using sklearn.decomposition.FastICA or MNE-ICA
            Reference: Jung et al. (2000) "Removing electroencephalographic artifacts by blind source separation"
        """
        logger.warning("ICA correction not yet implemented")
        return data
    
    def apply_imu_correction(
        self,
        eeg_data: np.ndarray,
        imu_data: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Apply IMU-based motion artifact correction (placeholder).
        
        For systems with integrated IMU sensors (accelerometer/gyroscope),
        motion artifacts can be regressed out using adaptive filtering.
        
        Args:
            eeg_data: EEG data
            imu_data: Synchronized IMU data (acceleration/rotation)
            
        Returns:
            np.ndarray: Motion-corrected EEG data
            
        Notes:
            TODO: Implement using adaptive LMS filtering or regression
            Reference: Sweeney et al. (2012) "A methodology for validating artifact removal techniques for physiological signals"
        """
        logger.warning("IMU-based correction not yet implemented")
        return eeg_data


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def check_data_quality(
    data: np.ndarray,
    fs: int = SAMPLING_RATE,
    verbose: bool = True
) -> dict:
    """
    Perform comprehensive data quality checks.
    
    Args:
        data: EEG data of shape (n_channels, n_samples)
        fs: Sampling frequency
        verbose: Print quality report
        
    Returns:
        dict: Quality metrics
    """
    metrics = {}
    
    # Check for NaN/Inf
    metrics['has_nan'] = np.any(np.isnan(data))
    metrics['has_inf'] = np.any(np.isinf(data))
    
    # Check amplitude range
    metrics['min_amplitude'] = np.min(data)
    metrics['max_amplitude'] = np.max(data)
    metrics['mean_amplitude'] = np.mean(data)
    metrics['std_amplitude'] = np.std(data)
    
    # Check for flatlines
    variance_per_channel = np.var(data, axis=-1)
    metrics['flatline_channels'] = np.sum(variance_per_channel < FLATLINE_THRESHOLD)
    
    # Check for extreme values
    metrics['extreme_values'] = np.sum(np.abs(data) > VOLTAGE_THRESHOLD)
    
    # Signal-to-noise ratio estimate (power in signal vs noise bands)
    freqs, psd = welch(data, fs=fs, axis=-1)
    signal_band = (freqs >= 1) & (freqs <= 45)
    noise_band = freqs > 45
    metrics['snr_db'] = 10 * np.log10(
        np.mean(psd[:, signal_band]) / (np.mean(psd[:, noise_band]) + 1e-10)
    )
    
    if verbose:
        print("=" * 60)
        print("EEG Data Quality Report")
        print("=" * 60)
        print(f"Shape: {data.shape}")
        print(f"NaN values: {metrics['has_nan']}")
        print(f"Inf values: {metrics['has_inf']}")
        print(f"Amplitude range: [{metrics['min_amplitude']:.2f}, {metrics['max_amplitude']:.2f}] µV")
        print(f"Mean amplitude: {metrics['mean_amplitude']:.2f} µV")
        print(f"Std amplitude: {metrics['std_amplitude']:.2f} µV")
        print(f"Flatline channels: {metrics['flatline_channels']}")
        print(f"Extreme values (>{VOLTAGE_THRESHOLD}µV): {metrics['extreme_values']}")
        print(f"Estimated SNR: {metrics['snr_db']:.2f} dB")
        print("=" * 60)
    
    return metrics


if __name__ == "__main__":
    # Demo and self-test
    print("EEG Preprocessing Module - Self Test")
    print("=" * 60)
    
    # Create test data: 32 channels, 5 seconds at 256 Hz
    np.random.seed(42)
    test_data = np.random.randn(32, 1280) * 10  # 10 µV noise
    
    # Add some EEG-like signals
    t = np.arange(1280) / 256
    alpha_wave = 15 * np.sin(2 * np.pi * 10 * t)  # 10 Hz alpha
    test_data[0] += alpha_wave
    
    print(f"Test data shape: {test_data.shape}")
    
    # Initialize preprocessor
    preprocessor = EEGPreprocessor()
    
    # Run quality check
    metrics = check_data_quality(test_data, verbose=True)
    
    # Preprocess
    print("\nRunning preprocessing pipeline...")
    clean_data, artifact_mask = preprocessor.preprocess(
        test_data,
        apply_notch=True,
        remove_dc=True,
        detect_artifacts=True
    )
    
    print(f"Clean data shape: {clean_data.shape}")
    print(f"Artifact-free samples: {np.sum(artifact_mask) / artifact_mask.size * 100:.1f}%")
    
    print("\nSelf-test complete!")
