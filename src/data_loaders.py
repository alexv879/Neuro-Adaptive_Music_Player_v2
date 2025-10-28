"""
Data Loaders for EEG Datasets - Neuro-Adaptive Music Player v2

This module provides unified data loading for multiple EEG emotion datasets
with support for DEAP, SEED, and simulated data generation.

Features:
    - DEAP dataset loading (.mat, .dat formats)
    - SEED dataset loading (.mat format)
    - Simulated EEG data generation for testing
    - Standardized output format: (n_trials, n_channels, n_samples) + labels
    - Data validation and quality checks
    - Caching for faster repeated loading
    - Metadata extraction (sampling rate, channel names, etc.)

Author: Alexandru Emanuel Vasile
License: Proprietary
Version: 2.0.0
"""

import os
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import warnings

import numpy as np

# Optional imports with graceful degradation
try:
    import scipy.io as sio
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not installed. .mat file loading disabled.")

try:
    import pyedflib
    PYEDFLIB_AVAILABLE = True
except ImportError:
    PYEDFLIB_AVAILABLE = False
    warnings.warn("pyedflib not installed. .edf file loading disabled.")

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False
    warnings.warn("h5py not installed. HDF5 .mat file loading disabled.")

try:
    import mat73
    MAT73_AVAILABLE = True
except ImportError:
    MAT73_AVAILABLE = False
    warnings.warn("mat73 not installed. MATLAB v7.3 .mat file loading may fail.")


# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class EEGDataset:
    """
    Container for EEG dataset with metadata.
    
    Attributes:
        data: EEG signals with shape (n_trials, n_channels, n_samples)
        labels: Emotion labels with shape (n_trials,) or (n_trials, n_dimensions)
        sampling_rate: Sampling frequency in Hz
        channel_names: List of channel names (e.g., ['Fp1', 'Fp2', ...])
        trial_durations: Duration of each trial in seconds
        metadata: Additional dataset-specific information
    """
    data: np.ndarray
    labels: np.ndarray
    sampling_rate: float
    channel_names: List[str]
    trial_durations: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Validate dataset after initialization."""
        # Validate shapes
        assert self.data.ndim == 3, f"Data must be 3D (trials, channels, samples), got {self.data.shape}"
        assert self.labels.ndim in [1, 2], f"Labels must be 1D or 2D, got {self.labels.shape}"
        assert len(self.data) == len(self.labels), f"Data and labels length mismatch: {len(self.data)} vs {len(self.labels)}"
        assert self.data.shape[1] == len(self.channel_names), f"Channel mismatch: {self.data.shape[1]} vs {len(self.channel_names)}"
        
        # Initialize metadata if None
        if self.metadata is None:
            self.metadata = {}
        
        # Calculate trial durations if not provided
        if self.trial_durations is None:
            self.trial_durations = np.full(len(self.data), self.data.shape[2] / self.sampling_rate)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get a single trial and its label."""
        return self.data[idx], self.labels[idx]
    
    def get_info(self) -> Dict[str, Any]:
        """Get dataset information."""
        return {
            "n_trials": len(self.data),
            "n_channels": self.data.shape[1],
            "n_samples_per_trial": self.data.shape[2],
            "sampling_rate": self.sampling_rate,
            "duration_per_trial": self.data.shape[2] / self.sampling_rate,
            "total_duration": len(self.data) * self.data.shape[2] / self.sampling_rate,
            "channel_names": self.channel_names,
            "label_shape": self.labels.shape,
            "data_dtype": self.data.dtype,
            "memory_mb": (self.data.nbytes + self.labels.nbytes) / (1024 ** 2)
        }
    
    def __repr__(self) -> str:
        info = self.get_info()
        return (f"EEGDataset(trials={info['n_trials']}, "
                f"channels={info['n_channels']}, "
                f"samples={info['n_samples_per_trial']}, "
                f"sr={info['sampling_rate']}Hz)")


class DEAPLoader:
    """
    Data loader for DEAP dataset (Database for Emotion Analysis using Physiological signals).
    
    Dataset Info:
        - 32 participants
        - 40 video/trial per participant
        - 32 EEG channels + 8 peripheral channels
        - 60-second trials (3 seconds baseline removed)
        - Labels: valence, arousal, dominance, liking (1-9 scale)
        
    Reference:
        Koelstra, S., et al. (2012). "DEAP: A Database for Emotion Analysis 
        using Physiological Signals." IEEE Transactions on Affective Computing.
        
    File Format:
        - Preprocessed: data_preprocessed_python/sXX.dat (pickle)
        - Raw: data_original/sXX.bdf (BioSemi format)
    
    Example:
        >>> loader = DEAPLoader(data_dir="data/DEAP/")
        >>> dataset = loader.load_subject(subject_id=1)
        >>> print(dataset.get_info())
    """
    
    def __init__(self, data_dir: str, preprocessed: bool = True):
        """
        Initialize DEAP data loader.
        
        Args:
            data_dir: Root directory containing DEAP dataset
            preprocessed: Whether to load preprocessed or raw data
        """
        self.data_dir = Path(data_dir)
        self.preprocessed = preprocessed
        
        if preprocessed:
            self.data_subdir = self.data_dir / "data_preprocessed_python"
        else:
            self.data_subdir = self.data_dir / "data_original"
        
        # DEAP channel names (32 EEG + 8 peripheral)
        self.eeg_channel_names = [
            'Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7',
            'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz',
            'Fp2', 'AF4', 'F4', 'F8', 'FC6', 'FC2', 'C4', 'T8',
            'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2', 'Fz', 'Cz'
        ]
        
        self.peripheral_channel_names = [
            'hEOG', 'vEOG', 'zEMG', 'tEMG', 'GSR', 'Respiration', 'Plethysmograph', 'Temperature'
        ]
        
        self.all_channel_names = self.eeg_channel_names + self.peripheral_channel_names
        
        logger.info(f"DEAPLoader initialized: {self.data_subdir}")
    
    def load_subject(
        self,
        subject_id: int,
        eeg_only: bool = True,
        label_type: str = "valence_arousal"
    ) -> EEGDataset:
        """
        Load data for a single subject.
        
        Args:
            subject_id: Subject ID (1-32)
            eeg_only: If True, return only 32 EEG channels (exclude peripheral)
            label_type: Label format - "valence_arousal", "all", or "binary"
            
        Returns:
            EEGDataset object with loaded data
        """
        if not 1 <= subject_id <= 32:
            raise ValueError(f"Invalid subject_id: {subject_id}. Must be 1-32.")
        
        # Load preprocessed pickle file
        filepath = self.data_subdir / f"s{subject_id:02d}.dat"
        
        if not filepath.exists():
            raise FileNotFoundError(f"DEAP file not found: {filepath}")
        
        logger.info(f"Loading DEAP subject {subject_id} from {filepath}")
        
        try:
            with open(filepath, 'rb') as f:
                subject_data = pickle.load(f, encoding='latin1')
            
            # Extract data and labels
            # Shape: (40 trials, 40 channels, 8064 samples) for preprocessed (128 Hz, 63 sec)
            data = subject_data['data']  # (40, 40, 8064)
            labels = subject_data['labels']  # (40, 4) - valence, arousal, dominance, liking
            
            # Select EEG channels only if requested
            if eeg_only:
                data = data[:, :32, :]  # Keep only first 32 channels (EEG)
                channel_names = self.eeg_channel_names
            else:
                channel_names = self.all_channel_names
            
            # Process labels
            if label_type == "valence_arousal":
                # Return valence and arousal (columns 0 and 1)
                processed_labels = labels[:, :2]
            elif label_type == "binary":
                # Binarize: high (>5) = 1, low (<=5) = 0
                valence_binary = (labels[:, 0] > 5).astype(int)
                arousal_binary = (labels[:, 1] > 5).astype(int)
                processed_labels = np.column_stack([valence_binary, arousal_binary])
            elif label_type == "all":
                processed_labels = labels  # All 4 dimensions
            else:
                raise ValueError(f"Unknown label_type: {label_type}")
            
            # Create dataset
            dataset = EEGDataset(
                data=data.astype(np.float32),
                labels=processed_labels.astype(np.float32),
                sampling_rate=128.0,  # DEAP preprocessed is 128 Hz
                channel_names=channel_names,
                metadata={
                    "dataset": "DEAP",
                    "subject_id": subject_id,
                    "label_type": label_type,
                    "n_trials": 40,
                    "trial_duration": 63.0  # seconds
                }
            )
            
            logger.info(f"Loaded DEAP subject {subject_id}: {dataset}")
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load DEAP subject {subject_id}: {e}")
            raise
    
    def load_all_subjects(
        self,
        eeg_only: bool = True,
        label_type: str = "valence_arousal",
        subjects: Optional[List[int]] = None
    ) -> EEGDataset:
        """
        Load and concatenate data from multiple subjects.
        
        Args:
            eeg_only: If True, return only EEG channels
            label_type: Label format
            subjects: List of subject IDs to load (default: all 32)
            
        Returns:
            Combined EEGDataset from all subjects
        """
        if subjects is None:
            subjects = list(range(1, 33))  # All 32 subjects
        
        all_data = []
        all_labels = []
        
        for subject_id in subjects:
            try:
                dataset = self.load_subject(subject_id, eeg_only, label_type)
                all_data.append(dataset.data)
                all_labels.append(dataset.labels)
            except Exception as e:
                logger.warning(f"Skipping subject {subject_id}: {e}")
                continue
        
        if not all_data:
            raise RuntimeError("No subjects could be loaded")
        
        # Concatenate all subjects
        combined_data = np.concatenate(all_data, axis=0)
        combined_labels = np.concatenate(all_labels, axis=0)
        
        dataset = EEGDataset(
            data=combined_data,
            labels=combined_labels,
            sampling_rate=128.0,
            channel_names=self.eeg_channel_names if eeg_only else self.all_channel_names,
            metadata={
                "dataset": "DEAP",
                "subjects": subjects,
                "n_subjects": len(subjects),
                "label_type": label_type
            }
        )
        
        logger.info(f"Loaded {len(subjects)} DEAP subjects: {dataset}")
        return dataset


class SEEDLoader:
    """
    Data loader for SEED dataset (SJTU Emotion EEG Dataset).
    
    Dataset Info:
        - 15 participants
        - 15 video clips per session (3 sessions per subject)
        - 62 EEG channels
        - ~4 minute trials
        - Labels: 3 emotions (positive=1, neutral=0, negative=-1)
        
    Reference:
        Zheng, W. L., & Lu, B. L. (2015). "Investigating Critical Frequency 
        Bands and Channels for EEG-based Emotion Recognition with Deep Neural Networks."
        
    File Format:
        - .mat files with preprocessed data
        - Sampling rate: 200 Hz (downsampled from 1000 Hz)
    
    Example:
        >>> loader = SEEDLoader(data_dir="data/SEED/")
        >>> dataset = loader.load_session(subject_id=1, session=1)
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize SEED data loader.
        
        Args:
            data_dir: Root directory containing SEED dataset
        """
        self.data_dir = Path(data_dir)
        
        # SEED 62-channel names (10-20 system)
        self.channel_names = [
            'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ',
            'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2',
            'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4',
            'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6',
            'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
            'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ',
            'O2', 'CB2'
        ]
        
        logger.info(f"SEEDLoader initialized: {self.data_dir}")
    
    def load_session(
        self,
        subject_id: int,
        session: int = 1
    ) -> EEGDataset:
        """
        Load data for one session of a subject.
        
        Args:
            subject_id: Subject ID (1-15)
            session: Session number (1-3)
            
        Returns:
            EEGDataset object
        """
        if not 1 <= subject_id <= 15:
            raise ValueError(f"Invalid subject_id: {subject_id}. Must be 1-15.")
        if not 1 <= session <= 3:
            raise ValueError(f"Invalid session: {session}. Must be 1-3.")
        
        # SEED filename format: typically like "1_20131027.mat", "1_20131030.mat", etc.
        # This varies by dataset version - adjust pattern as needed
        pattern = f"{subject_id}_*.mat"
        mat_files = sorted(self.data_dir.glob(pattern))
        
        if not mat_files:
            raise FileNotFoundError(f"No SEED files found matching: {self.data_dir / pattern}")
        
        if session > len(mat_files):
            raise ValueError(f"Session {session} not available for subject {subject_id}")
        
        filepath = mat_files[session - 1]
        logger.info(f"Loading SEED subject {subject_id} session {session} from {filepath}")
        
        try:
            # Try loading with scipy first
            if SCIPY_AVAILABLE:
                try:
                    mat_data = sio.loadmat(str(filepath))
                except NotImplementedError:
                    # File is MATLAB v7.3, need h5py or mat73
                    if H5PY_AVAILABLE:
                        mat_data = self._load_mat73_h5py(filepath)
                    elif MAT73_AVAILABLE:
                        mat_data = mat73.loadmat(str(filepath))
                    else:
                        raise ImportError("MATLAB v7.3 file detected. Install h5py or mat73.")
            else:
                raise ImportError("SciPy not available")
            
            # Extract trials (format varies by SEED version)
            # Common format: keys like 'de_LDS1', 'de_LDS2', etc. for differential entropy features
            # or raw data keys
            
            # Find data keys (exclude metadata keys like __header__, __version__)
            data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
            
            if not data_keys:
                raise ValueError(f"No data found in {filepath}")
            
            # Collect trials
            all_trials = []
            labels = []
            
            # SEED labels for 15 clips: -1 (negative), 0 (neutral), 1 (positive)
            # Repeated pattern: [-1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1]
            seed_labels = [-1, 0, 1] * 5
            
            for i, key in enumerate(sorted(data_keys)):
                trial_data = mat_data[key]
                
                # Ensure correct shape: (n_channels, n_samples) -> transpose if needed
                if trial_data.shape[0] == 62:
                    pass  # Already (channels, samples)
                elif trial_data.shape[1] == 62:
                    trial_data = trial_data.T  # Transpose to (channels, samples)
                else:
                    logger.warning(f"Unexpected shape for {key}: {trial_data.shape}")
                    continue
                
                all_trials.append(trial_data)
                
                # Assign label
                if i < len(seed_labels):
                    labels.append(seed_labels[i])
                else:
                    labels.append(0)  # Neutral for extra trials
            
            # Stack trials
            data = np.stack(all_trials, axis=0).astype(np.float32)  # (n_trials, 62, n_samples)
            labels_array = np.array(labels, dtype=np.int32)
            
            dataset = EEGDataset(
                data=data,
                labels=labels_array,
                sampling_rate=200.0,  # SEED is 200 Hz
                channel_names=self.channel_names,
                metadata={
                    "dataset": "SEED",
                    "subject_id": subject_id,
                    "session": session,
                    "n_trials": len(data)
                }
            )
            
            logger.info(f"Loaded SEED session: {dataset}")
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load SEED session: {e}")
            raise
    
    def _load_mat73_h5py(self, filepath: Path) -> Dict:
        """Load MATLAB v7.3 file using h5py."""
        import h5py
        
        mat_data = {}
        with h5py.File(str(filepath), 'r') as f:
            for key in f.keys():
                if key.startswith('__'):
                    continue
                mat_data[key] = np.array(f[key])
        
        return mat_data


class SimulatedEEGGenerator:
    """
    Generate simulated EEG data for testing and development.
    
    Creates realistic EEG signals with:
        - Frequency band components (delta, theta, alpha, beta, gamma)
        - Channel correlations
        - Artifacts (eye blinks, muscle noise)
        - Emotion-specific patterns
        
    Example:
        >>> generator = SimulatedEEGGenerator()
        >>> dataset = generator.generate(n_trials=100, emotion="happy")
        >>> print(dataset.get_info())
    """
    
    def __init__(
        self,
        sampling_rate: float = 256.0,
        n_channels: int = 32,
        duration: float = 10.0
    ):
        """
        Initialize simulated EEG generator.
        
        Args:
            sampling_rate: Sampling frequency in Hz
            n_channels: Number of EEG channels
            duration: Duration of each trial in seconds
        """
        self.sampling_rate = sampling_rate
        self.n_channels = n_channels
        self.duration = duration
        self.n_samples = int(duration * sampling_rate)
        
        # Generate channel names
        self.channel_names = [f"CH{i+1}" for i in range(n_channels)]
        
        # Frequency bands (Hz)
        self.bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }
        
        logger.info(f"SimulatedEEGGenerator initialized: {n_channels} channels @ {sampling_rate} Hz")
    
    def generate(
        self,
        n_trials: int = 100,
        emotion: Optional[str] = None,
        noise_level: float = 0.1
    ) -> EEGDataset:
        """
        Generate simulated EEG dataset.
        
        Args:
            n_trials: Number of trials to generate
            emotion: If specified, generate emotion-specific patterns
            noise_level: Amount of random noise (0-1)
            
        Returns:
            EEGDataset with simulated data
        """
        logger.info(f"Generating {n_trials} simulated trials (emotion={emotion})")
        
        data = np.zeros((n_trials, self.n_channels, self.n_samples), dtype=np.float32)
        
        # Generate each trial
        for trial in range(n_trials):
            for ch in range(self.n_channels):
                # Generate base signal with frequency components
                signal = self._generate_channel_signal(emotion)
                
                # Add channel-specific variations
                signal *= (0.8 + 0.4 * np.random.rand())  # Amplitude variation
                
                # Add noise
                signal += noise_level * np.random.randn(self.n_samples)
                
                # Occasionally add artifacts
                if np.random.rand() < 0.1:  # 10% chance
                    signal = self._add_artifact(signal)
                
                data[trial, ch, :] = signal
        
        # Generate labels based on emotion
        if emotion:
            # Single emotion label
            emotion_map = {"happy": 2, "calm": 1, "sad": 0, "angry": 3, "neutral": 1}
            label_value = emotion_map.get(emotion.lower(), 1)
            labels = np.full(n_trials, label_value, dtype=np.int32)
        else:
            # Random labels (5 classes)
            labels = np.random.randint(0, 5, n_trials, dtype=np.int32)
        
        dataset = EEGDataset(
            data=data,
            labels=labels,
            sampling_rate=self.sampling_rate,
            channel_names=self.channel_names,
            metadata={
                "dataset": "Simulated",
                "emotion": emotion,
                "noise_level": noise_level
            }
        )
        
        logger.info(f"Generated simulated dataset: {dataset}")
        return dataset
    
    def _generate_channel_signal(self, emotion: Optional[str] = None) -> np.ndarray:
        """Generate a single channel's signal with frequency components."""
        t = np.arange(self.n_samples) / self.sampling_rate
        signal = np.zeros(self.n_samples)
        
        # Define emotion-specific band power profiles
        if emotion == "happy":
            # High beta/gamma, moderate alpha
            band_powers = {'delta': 0.5, 'theta': 0.7, 'alpha': 1.0, 'beta': 1.5, 'gamma': 1.2}
        elif emotion == "calm":
            # High alpha, low beta/gamma
            band_powers = {'delta': 0.6, 'theta': 0.8, 'alpha': 1.5, 'beta': 0.5, 'gamma': 0.3}
        elif emotion == "sad":
            # High theta/alpha, low beta
            band_powers = {'delta': 0.8, 'theta': 1.2, 'alpha': 1.0, 'beta': 0.4, 'gamma': 0.3}
        elif emotion == "angry":
            # High beta/gamma, low alpha
            band_powers = {'delta': 0.5, 'theta': 0.6, 'alpha': 0.5, 'beta': 1.5, 'gamma': 1.5}
        else:
            # Neutral/balanced
            band_powers = {'delta': 0.7, 'theta': 0.8, 'alpha': 1.0, 'beta': 0.8, 'gamma': 0.6}
        
        # Generate signal with all frequency bands
        for band_name, (low_freq, high_freq) in self.bands.items():
            # Random frequency within band
            freq = low_freq + (high_freq - low_freq) * np.random.rand()
            
            # Band power
            power = band_powers[band_name]
            
            # Add sine wave with random phase
            phase = 2 * np.pi * np.random.rand()
            signal += power * np.sin(2 * np.pi * freq * t + phase)
        
        # Normalize
        signal = signal / (signal.std() + 1e-8)
        
        return signal
    
    def _add_artifact(self, signal: np.ndarray) -> np.ndarray:
        """Add random artifact to signal (eye blink or muscle noise)."""
        artifact_type = np.random.choice(['blink', 'muscle'])
        
        if artifact_type == 'blink':
            # Eye blink: brief high-amplitude spike
            blink_pos = np.random.randint(0, len(signal))
            blink_width = int(0.2 * self.sampling_rate)  # 200ms
            
            if blink_pos + blink_width < len(signal):
                # Gaussian-shaped artifact
                x = np.arange(blink_width)
                blink = 5 * np.exp(-((x - blink_width/2) ** 2) / (blink_width/4))
                signal[blink_pos:blink_pos+blink_width] += blink
        
        else:  # muscle
            # Muscle noise: high-frequency burst
            burst_start = np.random.randint(0, len(signal) // 2)
            burst_length = np.random.randint(int(0.5 * self.sampling_rate), int(2 * self.sampling_rate))
            burst_end = min(burst_start + burst_length, len(signal))
            
            noise = 2 * np.random.randn(burst_end - burst_start)
            signal[burst_start:burst_end] += noise
        
        return signal


# ==================== CONVENIENCE FUNCTIONS ====================

def load_deap(
    data_dir: str,
    subject_ids: Optional[Union[int, List[int]]] = None,
    eeg_only: bool = True,
    label_type: str = "valence_arousal"
) -> EEGDataset:
    """
    Convenience function to load DEAP dataset.
    
    Args:
        data_dir: Path to DEAP dataset directory
        subject_ids: Single subject ID or list of IDs (default: all 32)
        eeg_only: Return only EEG channels (exclude peripheral)
        label_type: "valence_arousal", "binary", or "all"
        
    Returns:
        EEGDataset with loaded data
        
    Example:
        >>> dataset = load_deap("data/DEAP/", subject_ids=[1, 2, 3])
        >>> print(f"Loaded {len(dataset)} trials")
    """
    loader = DEAPLoader(data_dir)
    
    if subject_ids is None:
        return loader.load_all_subjects(eeg_only=eeg_only, label_type=label_type)
    elif isinstance(subject_ids, int):
        return loader.load_subject(subject_ids, eeg_only=eeg_only, label_type=label_type)
    else:
        return loader.load_all_subjects(eeg_only=eeg_only, label_type=label_type, subjects=subject_ids)


def load_seed(
    data_dir: str,
    subject_id: int,
    session: int = 1
) -> EEGDataset:
    """
    Convenience function to load SEED dataset.
    
    Args:
        data_dir: Path to SEED dataset directory
        subject_id: Subject ID (1-15)
        session: Session number (1-3)
        
    Returns:
        EEGDataset with loaded data
        
    Example:
        >>> dataset = load_seed("data/SEED/", subject_id=1, session=1)
    """
    loader = SEEDLoader(data_dir)
    return loader.load_session(subject_id, session)


def generate_simulated_data(
    n_trials: int = 100,
    n_channels: int = 32,
    sampling_rate: float = 256.0,
    duration: float = 10.0,
    emotion: Optional[str] = None
) -> EEGDataset:
    """
    Generate simulated EEG data for testing.
    
    Args:
        n_trials: Number of trials
        n_channels: Number of EEG channels
        sampling_rate: Sampling frequency in Hz
        duration: Duration per trial in seconds
        emotion: Optional emotion for pattern generation
        
    Returns:
        EEGDataset with simulated data
        
    Example:
        >>> dataset = generate_simulated_data(n_trials=50, emotion="happy")
        >>> print(dataset.get_info())
    """
    generator = SimulatedEEGGenerator(sampling_rate, n_channels, duration)
    return generator.generate(n_trials, emotion)


# ==================== SELF-TEST ====================

if __name__ == "__main__":
    """Self-test and demonstration of data loaders."""
    
    print("=" * 60)
    print("EEG Data Loaders - Self Test")
    print("=" * 60)
    
    # Test 1: Simulated data generation
    print("\n[Test 1] Generating simulated EEG data...")
    dataset = generate_simulated_data(n_trials=20, emotion="happy")
    print(f"[OK] Generated dataset: {dataset}")
    print(f"  Info: {dataset.get_info()}")
    
    # Test 2: Test different emotions
    print("\n[Test 2] Testing emotion-specific patterns...")
    for emotion in ["happy", "sad", "calm", "angry"]:
        ds = generate_simulated_data(n_trials=5, emotion=emotion)
        print(f"  {emotion:8} -> mean={ds.data.mean():.3f}, std={ds.data.std():.3f}")
    
    # Test 3: Test dataset indexing
    print("\n[Test 3] Testing dataset indexing...")
    trial_data, label = dataset[0]
    print(f"  Trial shape: {trial_data.shape}")
    print(f"  Label: {label}")
    
    # Test 4: DEAP loader (if available)
    print("\n[Test 4] Testing DEAP loader...")
    deap_path = Path("data/DEAP/")
    if deap_path.exists():
        try:
            deap_ds = load_deap(str(deap_path), subject_ids=1)
            print(f"  [OK] Loaded DEAP: {deap_ds}")
        except Exception as e:
            print(f"  [FAIL] DEAP loading failed: {e}")
    else:
        print(f"  [SKIP] Skipped (DEAP data not found at {deap_path})")
    
    # Test 5: SEED loader (if available)
    print("\n[Test 5] Testing SEED loader...")
    seed_path = Path("data/SEED/")
    if seed_path.exists():
        try:
            seed_ds = load_seed(str(seed_path), subject_id=1, session=1)
            print(f"  [OK] Loaded SEED: {seed_ds}")
        except Exception as e:
            print(f"  [FAIL] SEED loading failed: {e}")
    else:
        print(f"  [SKIP] Skipped (SEED data not found at {seed_path})")
    
    print("\n" + "=" * 60)
    print("[DONE] Self-test complete!")
    print("=" * 60)
    print("\nTo use real datasets:")
    print("  1. Download DEAP: https://www.eecs.qmul.ac.uk/mmv/datasets/deap/")
    print("  2. Download SEED: https://bcmi.sjtu.edu.cn/home/seed/")
    print("  3. Place in data/DEAP/ and data/SEED/")
    print("  4. Run: python -m src.data_loaders")
