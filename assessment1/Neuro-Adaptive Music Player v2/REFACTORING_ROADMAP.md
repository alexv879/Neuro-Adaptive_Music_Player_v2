# Refactoring Roadmap
## Neuro-Adaptive Music Player v2 - Implementation Guide

**Version:** 1.0  
**Last Updated:** October 23, 2025  
**Estimated Effort:** 2-3 days for Phase 1, 1 week total

---

## Overview

This document provides a step-by-step plan to refactor the Neuro-Adaptive Music Player v2 codebase for improved readability, maintainability, and modularity. Each phase is designed to be completed independently without breaking existing functionality.

### Success Criteria
- ✅ All files under 300 lines
- ✅ All functions under 75 lines
- ✅ No code duplication
- ✅ Consistent logging (ASCII only, no emojis)
- ✅ Clear module boundaries
- ✅ 100% test pass rate maintained

---

## Phase 1: Critical Refactoring (Priority 1)

**Timeline:** 2-3 days  
**Goal:** Reduce file sizes, improve function readability

### Task 1.1: Split `eeg_features.py` (809 lines → 6 files)

**Files to Create:**
```
src/features/
├── __init__.py
├── extractor.py        # Main orchestrator (200 lines)
├── band_power.py       # Band power extraction (150 lines)
├── asymmetry.py        # FAA computation (100 lines)
├── statistical.py      # Statistical features (100 lines)
├── spectral.py         # Spectral features (100 lines)
└── windowing.py        # Window management (80 lines)
```

**Steps:**
1. Create directory: `mkdir src/features`
2. Create `__init__.py` with exports
3. Extract `BandPowerExtractor` class to `band_power.py`
4. Extract `FrontalAlphaAsymmetry` logic to `asymmetry.py`
5. Extract statistical functions to `statistical.py`
6. Extract spectral functions to `spectral.py`
7. Extract windowing logic to `windowing.py`
8. Create orchestrator in `extractor.py`
9. Update imports in dependent files
10. Run tests: `pytest tests/test_features.py -v`

**Verification:**
```bash
python -c "from src.features import EEGFeatureExtractor; print('Import OK')"
pytest tests/test_features.py -v
```

---

### Task 1.2: Split `emotion_recognition_model.py` (856 lines → 4 files)

**Files to Create:**
```
src/models/
├── __init__.py
├── emotion_model.py    # Model wrapper (150 lines)
├── architectures.py    # Network definitions (300 lines)
├── trainer.py          # Training logic (200 lines)
└── evaluator.py        # Evaluation metrics (150 lines)
```

**Steps:**
1. Create directory: `mkdir src/models`
2. Extract architecture builders to `architectures.py`
3. Extract training logic to `trainer.py`
4. Extract evaluation methods to `evaluator.py`
5. Keep main `EmotionRecognitionModel` wrapper in `emotion_model.py`
6. Update imports
7. Run tests: `pytest tests/test_models.py -v`

**Code Template for `architectures.py`:**
```python
"""Model architecture definitions."""

def create_cnn_bilstm_model(input_shape, n_classes):
    """Build CNN+BiLSTM hybrid model."""
    # ... (extract from _build_cnn_bilstm_model)

def create_cnn_model(input_shape, n_classes):
    """Build CNN-only model."""
    # ... (extract from _build_cnn_model)

def get_architecture_builder(name: str):
    """Factory to get architecture builder."""
    builders = {
        'cnn_bilstm': create_cnn_bilstm_model,
        'cnn': create_cnn_model,
        'bilstm': create_bilstm_model,
        'dense': create_dense_model
    }
    return builders[name]
```

---

### Task 1.3: Split `music_recommendation.py` (842 lines → 7 files)

**Files to Create:**
```
src/music/
├── __init__.py
├── engine.py           # Main orchestrator (150 lines)
├── spotify_player.py   # Spotify integration (200 lines)
├── youtube_player.py   # YouTube integration (150 lines)
├── local_player.py     # Local playback (100 lines)
├── emotion_mapper.py   # Emotion → genre (100 lines)
└── history.py          # History tracking (80 lines)
```

**Steps:**
1. Create directory: `mkdir src/music`
2. Extract `SpotifyPlayer` logic to `spotify_player.py`
3. Extract YouTube logic to `youtube_player.py`
4. Extract local file playback to `local_player.py`
5. Extract emotion-genre mapping to `emotion_mapper.py`
6. Extract `RecommendationHistory` to `history.py`
7. Create orchestrator in `engine.py`
8. Update imports
9. Run tests: `pytest tests/test_music.py -v`

---

### Task 1.4: Refactor Long Functions in `eeg_preprocessing.py`

**Functions to Refactor:**
- `preprocess()` - 85 lines → 30 lines + 6 helpers
- `detect_artifacts()` - 70 lines → 40 lines + 3 helpers

**Implementation:**

**File:** `src/preprocessing/preprocessor.py`

```python
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
        1. DC offset removal
        2. Bandpass filtering
        3. Notch filtering
        4. Artifact detection
        5. Bad channel interpolation
        6. Standardization
    
    Args:
        data: EEG data of shape (n_channels, n_samples) or (n_trials, n_channels, n_samples)
        apply_notch: Apply powerline notch filter
        remove_dc: Remove DC offset before filtering
        standardize: Standardize to zero mean and unit variance
        detect_artifacts: Return artifact mask
        interpolate_bad: Interpolate artifact-contaminated channels
    
    Returns:
        If detect_artifacts=False: preprocessed data
        If detect_artifacts=True: (preprocessed_data, artifact_mask)
    """
    # Validate input
    self._validate_input_data(data)
    
    # Apply preprocessing steps
    data = self._apply_dc_removal(data) if remove_dc else data
    data = self._apply_bandpass_filter(data)
    data = self._apply_notch_filter(data) if apply_notch else data
    
    # Handle artifacts
    artifact_mask = None
    if detect_artifacts or interpolate_bad:
        artifact_mask = self._detect_artifacts_pipeline(data)
        if interpolate_bad:
            data = self._interpolate_bad_channels_pipeline(data, artifact_mask)
    
    # Final standardization
    data = self._apply_standardization(data) if standardize else data
    
    logger.info(f"Preprocessing complete: {data.shape}")
    
    return (data, artifact_mask) if detect_artifacts else data


# Helper methods (add these to the class)

def _validate_input_data(self, data: np.ndarray) -> None:
    """Validate input data shape and values."""
    if data.size == 0:
        raise ValueError("Cannot preprocess empty array")
    if np.any(np.isnan(data)):
        raise ValueError("Input contains NaN values")
    if np.any(np.isinf(data)):
        raise ValueError("Input contains Inf values")
    logger.debug(f"Input validation passed: {data.shape}")


def _apply_dc_removal(self, data: np.ndarray) -> np.ndarray:
    """Remove DC offset and log."""
    data = self.remove_dc_offset(data, axis=-1)
    logger.debug("Removed DC offset")
    return data


def _apply_bandpass_filter(self, data: np.ndarray) -> np.ndarray:
    """Apply bandpass filter and log."""
    data = self.apply_bandpass(data, axis=-1)
    logger.debug(f"Applied bandpass filter: {self.bandpass_low}-{self.bandpass_high}Hz")
    return data


def _apply_notch_filter(self, data: np.ndarray) -> np.ndarray:
    """Apply notch filter and log."""
    data = self.apply_notch(data, axis=-1)
    logger.debug(f"Applied notch filter: {self.notch_freq}Hz")
    return data


def _detect_artifacts_pipeline(self, data: np.ndarray) -> np.ndarray:
    """Detect artifacts across all channels."""
    mask = self.detect_artifacts(data, method='all')
    n_artifacts = np.sum(~mask)
    logger.debug(f"Detected {n_artifacts} artifact samples")
    return mask


def _apply_standardization(self, data: np.ndarray) -> np.ndarray:
    """Standardize data and log."""
    data = self.standardize(data, axis=-1)
    logger.debug("Standardized data")
    return data


def _interpolate_bad_channels_pipeline(
    self, 
    data: np.ndarray, 
    artifact_mask: np.ndarray
) -> np.ndarray:
    """Identify and interpolate bad channels."""
    if data.ndim == 2:
        bad_channel_mask = np.mean(~artifact_mask, axis=-1) > 0.5
    else:
        bad_channel_mask = np.mean(~artifact_mask, axis=(0, 2)) > 0.5
    
    n_bad = np.sum(bad_channel_mask)
    if n_bad > 0:
        logger.info(f"Interpolating {n_bad} bad channels")
        if data.ndim == 2:
            data = self.interpolate_bad_channels(data, bad_channel_mask)
        else:
            for i in range(data.shape[0]):
                data[i] = self.interpolate_bad_channels(data[i], bad_channel_mask)
    
    return data
```

**Verification:**
```bash
pytest tests/test_preprocessing.py::test_preprocess -v
```

---

### Task 1.5: Add `examples/README.md`

**File:** `examples/README.md`

```markdown
# Examples Guide - Neuro-Adaptive Music Player v2

Learn the system progressively through these examples.

## 📚 Learning Path

Run examples in this order for best understanding:

### 1. Basic Preprocessing (10 minutes)
**File:** `01_basic_preprocessing.py`  
**What you'll learn:** EEG signal filtering, artifact detection, data quality

```bash
python examples/01_basic_preprocessing.py
```

**Output:** Preprocessed EEG signals, quality metrics

---

### 2. Feature Extraction (15 minutes)
**File:** `02_feature_extraction.py`  
**What you'll learn:** Band powers, frontal alpha asymmetry, statistical features

```bash
python examples/02_feature_extraction.py
```

**Output:** Feature vectors ready for model input

---

### 3. Emotion Recognition (30 minutes)
**File:** `03_emotion_recognition.py`  
**What you'll learn:** Train CNN+BiLSTM model, evaluate performance

```bash
# Using simulated data
python examples/03_emotion_recognition.py --dataset simulated

# Using DEAP dataset (requires download)
python examples/03_emotion_recognition.py --dataset deap --subject 1
```

**Output:** Trained model saved to `models/`, accuracy metrics

---

### 4. Music Recommendation (15 minutes)
**File:** `04_music_recommendation.py`  
**What you'll learn:** Emotion-to-music mapping, Spotify integration

```bash
# Using mock emotion
python examples/04_music_recommendation.py --emotion happy

# With Spotify (requires API key)
export SPOTIFY_CLIENT_ID="your_id"
export SPOTIFY_CLIENT_SECRET="your_secret"
python examples/04_music_recommendation.py --emotion happy --platform spotify
```

**Output:** Track recommendations, playback demonstration

---

### 5. LLM-Powered Recommendations (20 minutes)
**File:** `05_llm_recommendation.py`  
**What you'll learn:** Dynamic track suggestions using GPT-4

```bash
# Requires OpenAI API key
export OPENAI_API_KEY="sk-..."
python examples/05_llm_recommendation.py --mood "happy and energetic"
```

**Output:** AI-generated track recommendations with reasoning

---

### 6. Complete Pipeline (30 minutes)
**File:** `06_complete_pipeline.py`  
**What you'll learn:** Full EEG-to-music pipeline

```bash
# Simulated EEG → Emotion → LLM → Spotify
export OPENAI_API_KEY="sk-..."
export SPOTIFY_CLIENT_ID="your_id"
export SPOTIFY_CLIENT_SECRET="your_secret"

python examples/06_complete_pipeline.py \
  --mode simulated \
  --enable-spotify \
  --auto-play
```

**Output:** Real-time EEG processing with music playback

---

## 🎓 For Specific Use Cases

### Research: DEAP Dataset Analysis
```bash
python examples/03_emotion_recognition.py --dataset deap --all-subjects
```

### Development: Quick Testing
```bash
python examples/01_basic_preprocessing.py --quick
```

### Demo: Live Presentation
```bash
python examples/06_complete_pipeline.py --mode simulated --verbose
```

---

## 🔧 Troubleshooting

### "TensorFlow not available"
**Solution:** `pip install tensorflow>=2.10.0`

### "OpenAI API key not found"
**Solution:** `export OPENAI_API_KEY="sk-..."`  
Or create `.env` file with `OPENAI_API_KEY=sk-...`

### "Spotify credentials missing"
**Solution:** Get credentials from https://developer.spotify.com/dashboard  
Set `SPOTIFY_CLIENT_ID` and `SPOTIFY_CLIENT_SECRET` environment variables

---

## 📖 Next Steps

After completing all examples:
1. Read [ARCHITECTURE.md](../ARCHITECTURE.md) for system design
2. Review [API_REFERENCE.md](../docs/API_REFERENCE.md) for detailed API docs
3. See [CONTRIBUTING.md](../CONTRIBUTING.md) to add features

## 💡 Tips

- Start with simulated data before using real EEG datasets
- Use `--verbose` flag for detailed logging
- Check `logs/` directory for debug information
- Save models frequently during training (auto-checkpointing enabled)

---

**Questions?** Open an issue on GitHub or check the [FAQ](../docs/FAQ.md)
```

**Steps:**
1. Create file: `examples/README.md`
2. Verify all example files referenced exist
3. Test example commands in README

---

### Task 1.6: Verify Emoji/Symbol Cleanup

**Goal:** Ensure no emojis remain in logging or documentation

**Steps:**
1. Search for Unicode emojis:
   ```bash
   grep -rn "[\u2000-\uFFFD]" src/ examples/ --color
   ```

2. Check for specific emojis:
   ```bash
   grep -rn "✓\|⚠\|❌\|💡\|🎵\|⏱\|📖\|🔧\|⭐" src/ examples/
   ```

3. Replace any found with ASCII equivalents:
   - ✓ → `[OK]`
   - ⚠️ → `[WARNING]`
   - ❌ → `[ERROR]`
   - 💡 → `[TIP]`
   - 🎵 → `[MUSIC]`
   - ⏱ → `[TIME]`

4. Update logging standard in all files:
   ```python
   # Good
   logger.info("[OK] Operation completed")
   logger.warning("[WARNING] Missing optional dependency")
   logger.error("[FAIL] Operation failed: {reason}")
   
   # Bad
   logger.info("✓ Operation completed")  # NO EMOJI
   logger.warning("⚠️ Missing dependency")  # NO EMOJI
   ```

**Verification:**
```bash
# Should return no results
grep -rn "✓" src/
grep -rn "⚠" src/
grep -rn "❌" src/
```

---

## Phase 2: High Priority Refactoring (Priority 2)

**Timeline:** 2-3 days  
**Goal:** Eliminate duplication, improve utilities

### Task 2.1: Create Shared Utilities

**File:** `src/utils/validation.py`

```python
"""Input validation utilities."""

import numpy as np
from typing import Optional

def validate_array_not_empty(data: np.ndarray, context: str = "array") -> None:
    """
    Validate array is not empty.
    
    Args:
        data: Array to validate
        context: Description for error message
        
    Raises:
        ValueError: If array is empty
    """
    if data.size == 0:
        raise ValueError(f"Cannot process empty {context}")


def validate_array_shape(
    data: np.ndarray, 
    expected_ndim: int, 
    context: str = "array"
) -> None:
    """
    Validate array dimensionality.
    
    Args:
        data: Array to validate
        expected_ndim: Expected number of dimensions
        context: Description for error message
        
    Raises:
        ValueError: If shape doesn't match
    """
    if data.ndim != expected_ndim:
        raise ValueError(
            f"{context} must be {expected_ndim}D, "
            f"got {data.ndim}D with shape {data.shape}"
        )


def validate_array_values(data: np.ndarray, context: str = "array") -> None:
    """
    Validate array doesn't contain NaN or Inf.
    
    Args:
        data: Array to validate
        context: Description for error message
        
    Raises:
        ValueError: If array contains invalid values
    """
    if np.any(np.isnan(data)):
        raise ValueError(f"{context} contains NaN values")
    if np.any(np.isinf(data)):
        raise ValueError(f"{context} contains Inf values")


def validate_sampling_rate(fs: float, min_fs: float = 1.0, max_fs: float = 10000.0) -> None:
    """
    Validate sampling rate is reasonable.
    
    Args:
        fs: Sampling rate in Hz
        min_fs: Minimum allowed rate
        max_fs: Maximum allowed rate
        
    Raises:
        ValueError: If sampling rate is invalid
    """
    if not min_fs <= fs <= max_fs:
        raise ValueError(
            f"Sampling rate {fs}Hz is outside valid range [{min_fs}, {max_fs}]Hz"
        )


def validate_frequency_band(
    low: float, 
    high: float, 
    fs: float, 
    band_name: str = "band"
) -> None:
    """
    Validate frequency band parameters.
    
    Args:
        low: Lower frequency bound
        high: Upper frequency bound
        fs: Sampling rate
        band_name: Band description for error message
        
    Raises:
        ValueError: If band parameters are invalid
    """
    if low >= high:
        raise ValueError(f"{band_name}: low ({low}) must be < high ({high})")
    
    nyquist = fs / 2
    if high >= nyquist:
        raise ValueError(
            f"{band_name}: high ({high}Hz) must be < Nyquist ({nyquist}Hz)"
        )
    
    if low <= 0:
        raise ValueError(f"{band_name}: low ({low}Hz) must be positive")
```

**File:** `src/utils/array_utils.py`

```python
"""Array manipulation utilities."""

import numpy as np
from typing import Tuple

def ensure_3d(data: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Ensure data is 3D (n_trials, n_channels, n_samples).
    
    Args:
        data: Input array (2D or 3D)
        
    Returns:
        Tuple of (3d_data, was_2d_originally)
        
    Raises:
        ValueError: If data is not 2D or 3D
    """
    if data.ndim == 2:
        return data[np.newaxis, :, :], True
    elif data.ndim == 3:
        return data, False
    else:
        raise ValueError(f"Expected 2D or 3D array, got {data.ndim}D")


def restore_original_shape(data: np.ndarray, was_2d: bool) -> np.ndarray:
    """
    Restore original shape if data was 2D.
    
    Args:
        data: 3D array
        was_2d: Whether original data was 2D
        
    Returns:
        Original shape (2D if was_2d=True, otherwise unchanged)
    """
    if was_2d and data.ndim == 3 and data.shape[0] == 1:
        return data[0]
    return data


def safe_divide(numerator: np.ndarray, denominator: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    Safely divide arrays avoiding division by zero.
    
    Args:
        numerator: Numerator array
        denominator: Denominator array
        epsilon: Small value to add to denominator
        
    Returns:
        numerator / (denominator + epsilon)
    """
    return numerator / (denominator + epsilon)


def normalize_array(data: np.ndarray, axis: int = -1, method: str = 'zscore') -> np.ndarray:
    """
    Normalize array using specified method.
    
    Args:
        data: Input array
        axis: Axis along which to normalize
        method: 'zscore' (z-score), 'minmax' (0-1), or 'l2' (unit norm)
        
    Returns:
        Normalized array
    """
    if method == 'zscore':
        mean = np.mean(data, axis=axis, keepdims=True)
        std = np.std(data, axis=axis, keepdims=True)
        return safe_divide(data - mean, std)
    
    elif method == 'minmax':
        min_val = np.min(data, axis=axis, keepdims=True)
        max_val = np.max(data, axis=axis, keepdims=True)
        return safe_divide(data - min_val, max_val - min_val)
    
    elif method == 'l2':
        norm = np.linalg.norm(data, axis=axis, keepdims=True)
        return safe_divide(data, norm)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
```

**Steps:**
1. Create `src/utils/` directory
2. Add `__init__.py`, `validation.py`, `array_utils.py`
3. Update all files to use shared utilities
4. Remove duplicate validation code
5. Run all tests: `pytest tests/ -v`

---

### Task 2.2: Extract LLM Prompt Templates

**File:** `src/llm/prompts.py`

```python
"""LLM prompt templates for music recommendation."""

from typing import Dict, Any
import textwrap

class PromptBuilder:
    """Build LLM prompts for music recommendation."""
    
    BASIC_TEMPLATE = textwrap.dedent("""
        You are a music recommendation expert. The user is currently feeling: {mood_tag}
        Confidence in emotion detection: {confidence:.0%}
        
        Recommend {n_tracks} songs that match this mood. Format each as:
        Artist - Title: Brief reasoning
        
        Songs:
    """).strip()
    
    DETAILED_TEMPLATE = textwrap.dedent("""
        You are a music recommendation expert with deep knowledge of music psychology.
        
        **EEG-Detected Mood:** {mood_tag} {confidence_text} ({confidence:.0%})
        
        Based on this emotional state, recommend {n_tracks} songs that will:
        - Resonate with the current mood
        - Provide appropriate emotional support or enhancement
        - Match the detected arousal and valence levels
        
        Format each recommendation as:
        **Artist - Title**
        Reasoning: (1-2 sentences explaining why this song fits)
        
        Songs:
    """).strip()
    
    CONTEXTUAL_TEMPLATE = textwrap.dedent("""
        You are a music recommendation expert with knowledge of music psychology and context-awareness.
        
        **User Context:**
        - Current Mood: {mood_tag}
        - Detection Confidence: {confidence:.0%}
        - Time of Day: {time_of_day}
        - Activity: {activity}
        {additional_context}
        
        Recommend {n_tracks} songs that:
        1. Match the detected mood ({mood_tag})
        2. Fit the current context (time: {time_of_day}, activity: {activity})
        3. Provide emotional support or enhancement
        
        Consider:
        - Energy level appropriate for time/activity
        - Lyrics that resonate with emotional state
        - Tempo and rhythm matching arousal level
        
        Format: **Artist - Title**
        Reasoning: (why this song fits the context and mood)
        
        Songs:
    """).strip()
    
    def build(
        self, 
        template: str, 
        mood_tag: str, 
        confidence: float, 
        n_tracks: int = 3,
        **kwargs
    ) -> str:
        """
        Build prompt from template with variables.
        
        Args:
            template: Template name ('basic', 'detailed', 'contextual')
            mood_tag: Descriptive mood string
            confidence: Detection confidence (0-1)
            n_tracks: Number of tracks to recommend
            **kwargs: Additional context variables
            
        Returns:
            Formatted prompt string
        """
        # Build context dictionary
        confidence_text = self._get_confidence_text(confidence)
        
        context = {
            'mood_tag': mood_tag,
            'confidence': confidence,
            'confidence_text': confidence_text,
            'n_tracks': n_tracks,
            'time_of_day': kwargs.get('time_of_day', 'unknown'),
            'activity': kwargs.get('activity', 'unknown'),
            'additional_context': self._format_additional_context(kwargs)
        }
        
        # Select template
        if template == 'basic':
            return self.BASIC_TEMPLATE.format(**context)
        elif template == 'detailed':
            return self.DETAILED_TEMPLATE.format(**context)
        elif template == 'contextual':
            return self.CONTEXTUAL_TEMPLATE.format(**context)
        else:
            raise ValueError(f"Unknown template: {template}")
    
    def _get_confidence_text(self, confidence: float) -> str:
        """Convert confidence to descriptive text."""
        if confidence >= 0.9:
            return "(very confident)"
        elif confidence >= 0.75:
            return "(confident)"
        elif confidence >= 0.6:
            return "(moderately confident)"
        else:
            return "(less confident)"
    
    def _format_additional_context(self, kwargs: Dict[str, Any]) -> str:
        """Format additional context parameters."""
        extras = []
        if 'weather' in kwargs:
            extras.append(f"- Weather: {kwargs['weather']}")
        if 'location' in kwargs:
            extras.append(f"- Location: {kwargs['location']}")
        if 'previous_tracks' in kwargs:
            extras.append(f"- Recently Played: {', '.join(kwargs['previous_tracks'][:3])}")
        
        return "\n".join(extras) if extras else ""
```

**Steps:**
1. Create `src/llm/prompts.py`
2. Update `LLMMusicRecommender._build_prompt()` to use `PromptBuilder`
3. Add tests for prompt generation
4. Verify LLM recommendations still work

---

### Task 2.3: Split `config.py` by Concern

**New Structure:**
```
src/config/
├── __init__.py          # Export all configs
├── signal_config.py     # EEG signal parameters
├── model_config.py      # Model hyperparameters
├── paths_config.py      # File paths
└── logging_config.py    # Logging settings
```

**File:** `src/config/__init__.py`

```python
"""Configuration module - exports all config classes."""

from .signal_config import SignalConfig
from .model_config import ModelConfig
from .paths_config import PathsConfig
from .logging_config import LoggingConfig

# Backward compatibility - create combined Config class
class Config:
    """Combined configuration (backward compatible)."""
    def __init__(self):
        self.signal = SignalConfig()
        self.model = ModelConfig()
        self.paths = PathsConfig()
        self.logging = LoggingConfig()
        
        # Legacy attributes (for backward compatibility)
        self.SAMPLING_RATE = self.signal.SAMPLING_RATE
        self.BANDPASS_LOWCUT = self.signal.BANDPASS_LOWCUT
        self.BANDPASS_HIGHCUT = self.signal.BANDPASS_HIGHCUT
        # ... (add all backward-compatible attributes)

__all__ = ['Config', 'SignalConfig', 'ModelConfig', 'PathsConfig', 'LoggingConfig']
```

**Migration Path:**
1. Keep `config.py` for now (backward compatibility)
2. Create new config package
3. Update new code to use specific configs: `from src.config import SignalConfig`
4. Gradually migrate old code
5. Eventually deprecate `config.py`

---

### Task 2.4: Clean Up Obsolete Files

**Files to Review:**
```
cleanup_emojis.py          → DELETE (served its purpose)
demo_output.txt            → MOVE to examples/outputs/
llm_demo_output.txt        → MOVE to examples/outputs/
CLEANUP_SUMMARY.md         → MOVE to docs/archive/
REFACTORING_LOG.md         → MOVE to docs/archive/
```

**Steps:**
1. Create `examples/outputs/` directory
2. Move demo outputs
3. Create `docs/archive/` directory
4. Archive old documentation
5. Delete `cleanup_emojis.py`
6. Update `.gitignore` to exclude `examples/outputs/`

---

## Phase 3: Nice to Have (Priority 3)

**Timeline:** 3-4 days  
**Goal:** Documentation, developer experience

### Task 3.1: Generate API Documentation

**Tool:** Sphinx

**Steps:**
1. Install Sphinx: `pip install sphinx sphinx-rtd-theme`
2. Initialize: `sphinx-quickstart docs/sphinx`
3. Configure `docs/sphinx/conf.py`:
   ```python
   extensions = [
       'sphinx.ext.autodoc',
       'sphinx.ext.napoleon',  # For NumPy/Google docstrings
       'sphinx.ext.viewcode',
   ]
   html_theme = 'sphinx_rtd_theme'
   ```
4. Create API docs: `sphinx-apidoc -o docs/sphinx/source src/`
5. Build: `cd docs/sphinx && make html`
6. Add to CI/CD: Auto-build on push

---

### Task 3.2: Create Developer Guide

**File:** `docs/DEVELOPER_GUIDE.md`

```markdown
# Developer Guide - Neuro-Adaptive Music Player v2

## Getting Started

### Setup Development Environment

1. Clone repository:
   ```bash
   git clone https://github.com/alexv879/Neuro-Adaptive_Music_Player_v2.git
   cd Neuro-Adaptive_Music_Player_v2
   ```

2. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Coding Standards

### Python Style Guide

- Follow PEP 8
- Use Black for formatting: `black src/`
- Use isort for imports: `isort src/`
- Type hints required for all functions
- Docstrings required (NumPy style)

### Naming Conventions

- Classes: `PascalCase` (e.g., `EEGPreprocessor`)
- Functions: `snake_case` (e.g., `extract_features`)
- Constants: `UPPER_CASE` (e.g., `SAMPLING_RATE`)
- Private methods: `_leading_underscore` (e.g., `_validate_input`)

### File Length Guidelines

- Maximum 300 lines per file
- Maximum 75 lines per function
- If exceeded, refactor into smaller modules

### Logging Standards

Use ASCII markers, no emojis:
```python
logger.info("[OK] Operation completed")
logger.warning("[WARNING] Potential issue")
logger.error("[FAIL] Operation failed")
logger.debug("[DEBUG] Detailed info")
```

## Adding New Features

### 1. Preprocessing Feature

**File:** `src/preprocessing/your_feature.py`

```python
"""Your new preprocessing feature."""

import numpy as np
from ..utils.validation import validate_array_not_empty

def your_preprocessing_function(data: np.ndarray) -> np.ndarray:
    """
    Description of what this does.
    
    Args:
        data: Input data
        
    Returns:
        Processed data
    """
    validate_array_not_empty(data, "input data")
    # Your implementation
    return processed_data
```

**Test:** `tests/test_your_feature.py`

### 2. Feature Extraction Method

**File:** `src/features/your_feature.py`

```python
"""Your new feature extractor."""

class YourFeatureExtractor:
    """Extract your custom features."""
    
    def extract(self, data: np.ndarray) -> np.ndarray:
        """Extract features."""
        # Implementation
```

### 3. Model Architecture

**File:** `src/models/architectures.py`

```python
def create_your_model(input_shape, n_classes):
    """Build your custom architecture."""
    # Implementation
```

## Testing

### Run Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_preprocessing.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

### Writing Tests

```python
"""Test your feature."""

import pytest
import numpy as np
from src.your_module import YourClass

def test_your_feature():
    """Test basic functionality."""
    # Arrange
    data = np.random.randn(32, 1280)
    processor = YourClass()
    
    # Act
    result = processor.process(data)
    
    # Assert
    assert result.shape == data.shape
    assert not np.any(np.isnan(result))
```

## Documentation

### Docstring Format (NumPy Style)

```python
def function_name(param1: type1, param2: type2) -> return_type:
    """
    Short one-line summary.
    
    Longer description explaining what the function does,
    its purpose, and any important details.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When this exception is raised
        
    Example:
        >>> result = function_name(arg1, arg2)
        >>> print(result)
        Expected output
    """
```

## Git Workflow

### Branching Strategy

- `main` - Production-ready code
- `develop` - Integration branch
- `feature/your-feature` - New features
- `bugfix/issue-number` - Bug fixes

### Commit Messages

Format:
```
type(scope): Short description

Longer description if needed

Fixes #issue_number
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `refactor`: Code restructuring
- `test`: Test additions/changes
- `chore`: Maintenance tasks

Example:
```
feat(preprocessing): Add adaptive filtering method

Implemented adaptive filter based on LMS algorithm for
better noise cancellation in high-artifact conditions.

Fixes #123
```

## CI/CD Pipeline

Our CI runs on every push:

1. Linting (black, isort, flake8)
2. Type checking (mypy)
3. Unit tests (pytest)
4. Coverage check (>80%)
5. Build documentation
6. Integration tests

## Performance Guidelines

- Use vectorized NumPy operations (avoid loops)
- Profile code with cProfile for bottlenecks
- Target <50ms for real-time processing functions
- Document algorithmic complexity in docstrings

## Getting Help

- Check [FAQ](FAQ.md)
- Search [Issues](https://github.com/alexv879/Neuro-Adaptive_Music_Player_v2/issues)
- Ask on Discussions

## Code Review Checklist

Before submitting PR:
- [ ] Tests pass locally
- [ ] New tests added for new features
- [ ] Docstrings added/updated
- [ ] Type hints added
- [ ] No functions >75 lines
- [ ] No files >300 lines
- [ ] Logging uses ASCII (no emojis)
- [ ] Code formatted with Black
- [ ] Imports sorted with isort
```

---

## Verification Checklist

After each phase, verify:

### Phase 1 Verification
- [ ] All files <300 lines
- [ ] All functions <75 lines
- [ ] No emojis in logging
- [ ] All tests passing
- [ ] `examples/README.md` created
- [ ] Imports work: `python -c "from src.features import EEGFeatureExtractor; print('OK')"`

### Phase 2 Verification
- [ ] Shared utilities created and used
- [ ] No duplicate validation code
- [ ] Prompt templates extracted
- [ ] Obsolete files cleaned up
- [ ] All tests still passing

### Phase 3 Verification
- [ ] API docs generated
- [ ] Developer guide complete
- [ ] Pre-commit hooks working
- [ ] CI/CD pipeline passing

---

## Timeline Summary

| Phase | Duration | Key Tasks |
|-------|----------|-----------|
| Phase 1 | 2-3 days | Split large files, refactor functions, add examples README |
| Phase 2 | 2-3 days | Create utilities, extract templates, clean up |
| Phase 3 | 3-4 days | Documentation, developer tools |
| **Total** | **7-10 days** | Complete refactoring |

---

## Success Metrics

**Before Refactoring:**
- Average file size: 750 lines
- Longest function: 120 lines
- Code duplication: ~15%
- Import clarity: 60%

**After Refactoring:**
- Average file size: <200 lines ✅
- Longest function: <75 lines ✅
- Code duplication: <5% ✅
- Import clarity: >90% ✅

---

**Questions?** Refer to [READABILITY_ANALYSIS.md](READABILITY_ANALYSIS.md) for detailed rationale.

**Version:** 1.0  
**Last Updated:** October 23, 2025  
**Maintained By:** Development Team
