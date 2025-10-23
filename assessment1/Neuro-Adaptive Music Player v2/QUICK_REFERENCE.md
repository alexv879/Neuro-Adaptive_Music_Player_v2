# Quick Reference: Code Quality Improvements
## Neuro-Adaptive Music Player v2

**Last Updated:** October 23, 2025

---

## 🎯 Top 5 Immediate Actions

### 1. Split Long Files ⚠️ CRITICAL
```bash
# Current: 6 files >700 lines
# Target: All files <300 lines

Priority files:
- eeg_features.py (809 lines) → split into features/ package
- emotion_recognition_model.py (856 lines) → split into models/ package
- music_recommendation.py (842 lines) → split into music/ package
```

### 2. Refactor Long Functions ⚠️ CRITICAL
```python
# Functions >75 lines need refactoring:

eeg_preprocessing.py:
  - preprocess() - 85 lines → Extract helpers

emotion_recognition_model.py:
  - _build_cnn_bilstm_model() - 120 lines → Separate file

llm_music_recommender.py:
  - _build_prompt() - 110 lines → Extract to prompts.py
```

### 3. Add examples/README.md ✅ HIGH PRIORITY
Guide users through examples in logical order with clear learning objectives.

### 4. Create Shared Utilities ✅ HIGH PRIORITY
```python
# Eliminate duplication:
src/utils/
  ├── validation.py  # Input validation
  └── array_utils.py # Array manipulation
```

### 5. Verify Emoji Cleanup ✅ VERIFY
```bash
# Check for remaining emojis
grep -rn "[\u2000-\uFFFD]" src/
```

---

## 📏 Code Standards Quick Reference

### File Length
- **Maximum:** 300 lines
- **Ideal:** <200 lines
- **Action:** Split into submodules if exceeded

### Function Length
- **Maximum:** 75 lines
- **Ideal:** <50 lines
- **Action:** Extract helper methods

### Class Length
- **Maximum:** 400 lines
- **Ideal:** <300 lines
- **Action:** Use composition pattern

---

## 📝 Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Classes | PascalCase | `EEGPreprocessor` |
| Functions | snake_case | `extract_features()` |
| Constants | UPPER_SNAKE | `SAMPLING_RATE` |
| Private | _leading_underscore | `_validate_input()` |
| Modules | lowercase | `preprocessing.py` |
| Packages | lowercase | `features/` |

---

## 🪵 Logging Standards

### ✅ Good (ASCII only)
```python
logger.info("[OK] Operation completed successfully")
logger.info("[DONE] Processing finished")
logger.warning("[WARNING] Missing optional dependency")
logger.error("[FAIL] Operation failed: {reason}")
logger.debug("[DEBUG] Variable value: {value}")
```

### ❌ Bad (No emojis!)
```python
logger.info("✓ Operation completed")  # NO
logger.warning("⚠️ Missing dependency")  # NO
logger.error("❌ Failed")  # NO
```

### Emoji → ASCII Mapping
| Emoji | ASCII |
|-------|-------|
| ✓ | `[OK]` |
| ⚠️ | `[WARNING]` |
| ❌ | `[ERROR]` or `[FAIL]` |
| 💡 | `[TIP]` |
| 🎵 | `[MUSIC]` |
| ⏱ | `[TIME]` |
| 📖 | `[DOC]` |
| 🔧 | `[DEBUG]` |

---

## 📚 Documentation Standards

### File-Level Docstring
```python
"""
Module Name - Brief Description
================================

Longer description of module purpose, features, and usage.

Key Features:
- Feature 1
- Feature 2

References:
- Paper citation 1
- Paper citation 2

Author: Name
License: Proprietary
Version: X.Y.Z
"""
```

### Class Docstring (NumPy Style)
```python
class ClassName:
    """
    One-line summary of class purpose.
    
    Longer description explaining what the class does,
    its role in the system, and key design decisions.
    
    Attributes:
        attr1 (type): Description of attribute 1
        attr2 (type): Description of attribute 2
    
    Example:
        >>> obj = ClassName(param1, param2)
        >>> result = obj.method()
        >>> print(result)
        expected output
    """
```

### Function Docstring (NumPy Style)
```python
def function_name(
    param1: type1,
    param2: type2,
    optional_param: type3 = default
) -> return_type:
    """
    One-line summary of function purpose.
    
    Longer description explaining what the function does,
    algorithm used, and any important implementation details.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        optional_param: Description (default: {default})
    
    Returns:
        Description of return value, including shape if np.ndarray
    
    Raises:
        ValueError: When and why this is raised
        TypeError: When and why this is raised
    
    Example:
        >>> result = function_name(arg1, arg2)
        >>> print(result.shape)
        (32, 128)
    
    Note:
        Additional notes, warnings, or references
    """
```

---

## 🧪 Testing Standards

### Test Function Template
```python
def test_feature_name():
    """Test description following Arrange-Act-Assert pattern."""
    # Arrange: Set up test data
    input_data = np.random.randn(32, 1280)
    processor = YourClass()
    
    # Act: Execute the function
    result = processor.process(input_data)
    
    # Assert: Verify expectations
    assert result.shape == input_data.shape
    assert not np.any(np.isnan(result))
    assert result.dtype == np.float64
```

### Test Coverage Targets
- **Minimum:** 80%
- **Target:** 90%
- **Critical modules:** 95%+ (preprocessing, features, models)

---

## 🗂️ Import Organization

### Order (use isort)
```python
# 1. Standard library
import os
import sys
from typing import Dict, List, Optional

# 2. Third-party
import numpy as np
import tensorflow as tf
from scipy.signal import welch

# 3. Local application
from src.config import Config
from src.utils.validation import validate_array_not_empty
from .helpers import helper_function
```

### Import Style
```python
# ✅ Good
from src.features import EEGFeatureExtractor
from src.preprocessing import EEGPreprocessor

# ❌ Avoid
from src.features import *  # Too broad
import src.features as f  # Unclear alias
```

---

## 🔧 Refactoring Patterns

### Pattern 1: Extract Method
**When:** Function >75 lines or has multiple responsibilities

**Before:**
```python
def process(data):
    # Validate input
    if data.size == 0:
        raise ValueError("Empty")
    
    # Preprocess
    data = data - np.mean(data)
    
    # Filter
    filtered = apply_filter(data)
    
    # Extract features
    features = []
    for band in bands:
        power = compute_power(filtered, band)
        features.append(power)
    
    return np.array(features)
```

**After:**
```python
def process(data):
    """Process data through full pipeline."""
    self._validate_input(data)
    data = self._preprocess(data)
    filtered = self._apply_filter(data)
    features = self._extract_features(filtered)
    return features

def _validate_input(self, data):
    """Validate input data."""
    if data.size == 0:
        raise ValueError("Empty")

def _preprocess(self, data):
    """Remove DC offset."""
    return data - np.mean(data)

# ... etc
```

### Pattern 2: Composition Over Inheritance
**When:** Class >400 lines with multiple concerns

**Before:**
```python
class MusicEngine:
    def search_spotify(self):
        # 100 lines
    
    def search_youtube(self):
        # 100 lines
    
    def play_local(self):
        # 100 lines
```

**After:**
```python
class MusicEngine:
    def __init__(self):
        self.spotify = SpotifyPlayer()
        self.youtube = YouTubePlayer()
        self.local = LocalPlayer()
    
    def search(self, platform):
        if platform == 'spotify':
            return self.spotify.search()
        # ...

# Separate files
class SpotifyPlayer:
    def search(self):
        # 50 lines
```

### Pattern 3: Strategy Pattern
**When:** Conditional logic based on type/method

**Before:**
```python
def extract_power(data, method):
    if method == 'welch':
        # 30 lines of Welch logic
    elif method == 'fft':
        # 30 lines of FFT logic
```

**After:**
```python
class WelchExtractor:
    def extract(self, data):
        # 30 lines

class FFTExtractor:
    def extract(self, data):
        # 30 lines

extractors = {
    'welch': WelchExtractor(),
    'fft': FFTExtractor()
}
power = extractors[method].extract(data)
```

---

## 🚀 Performance Guidelines

### Prefer Vectorization
```python
# ❌ Slow (loops)
result = []
for i in range(len(data)):
    result.append(data[i] ** 2)

# ✅ Fast (vectorized)
result = data ** 2
```

### Memory Efficiency
```python
# ✅ Use views instead of copies
windowed = np.lib.stride_tricks.as_strided(...)

# ✅ Process in chunks for large data
for chunk in data_chunks:
    process(chunk)
```

### Profiling
```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()
# Your code
profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10 functions
```

---

## 🐛 Common Issues & Solutions

### Issue: "Import numpy could not be resolved"
**Solution:**
```bash
pip install numpy scipy
```

### Issue: "Function too long"
**Solution:** Extract helper methods, see refactoring patterns above

### Issue: "File too large"
**Solution:** Split into package:
```python
# Before: my_module.py (800 lines)
# After:
my_module/
  ├── __init__.py  # Exports
  ├── core.py      # Main class
  ├── helpers.py   # Helper functions
  └── utils.py     # Utilities
```

### Issue: "Code duplication"
**Solution:** Create shared utilities in `src/utils/`

---

## ✅ Pre-Commit Checklist

Before every commit:
- [ ] Run Black: `black src/`
- [ ] Run isort: `isort src/`
- [ ] Run tests: `pytest tests/ -v`
- [ ] Check coverage: `pytest --cov=src --cov-report=term-missing`
- [ ] Verify no long files: `find src/ -name "*.py" -exec wc -l {} \; | sort -rn | head -10`
- [ ] Verify no long functions: Use IDE or grep
- [ ] Check for emojis: `grep -rn "[\u2000-\uFFFD]" src/`
- [ ] Update docstrings if needed
- [ ] Add/update tests for new features

---

## 📖 Additional Resources

- **Full Analysis:** [READABILITY_ANALYSIS.md](READABILITY_ANALYSIS.md)
- **Detailed Roadmap:** [REFACTORING_ROADMAP.md](REFACTORING_ROADMAP.md)
- **Architecture:** [ARCHITECTURE.md](ARCHITECTURE.md)
- **Contributing:** [CONTRIBUTING.md](CONTRIBUTING.md)

---

## 🎓 Quick Commands

```bash
# Format code
black src/ examples/ tests/
isort src/ examples/ tests/

# Run tests
pytest tests/ -v
pytest tests/ --cov=src --cov-report=html

# Check file lengths
find src/ -name "*.py" -exec wc -l {} \; | sort -rn | head -10

# Find long functions (crude but works)
grep -n "^def \|^class " src/**/*.py | grep -v "__"

# Search for emojis
grep -rn "[\u2000-\uFFFD]" src/

# Count lines of code
cloc src/

# Profile performance
python -m cProfile -s cumulative examples/your_example.py
```

---

**Quick Reference Version:** 1.0  
**For detailed guidance, see:** REFACTORING_ROADMAP.md  
**Last Updated:** October 23, 2025
