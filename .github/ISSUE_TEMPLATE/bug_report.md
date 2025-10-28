---
name: Bug Report
about: Report a bug or unexpected behavior
title: '[BUG] '
labels: bug
assignees: ''
---

## Bug Description
A clear and concise description of what the bug is.

## To Reproduce
Steps to reproduce the behavior:
1. Import module '...'
2. Call function '...'
3. With parameters '...'
4. See error

## Expected Behavior
A clear and concise description of what you expected to happen.

## Actual Behavior
What actually happened instead.

## Error Message
```python
# Paste full error traceback here
```

## Environment
- **OS**: [e.g., Windows 11, Ubuntu 22.04, macOS 13]
- **Python Version**: [e.g., 3.9.7]
- **TensorFlow Version**: [e.g., 2.10.0]
- **NumPy Version**: [e.g., 1.23.0]
- **Installation Method**: [e.g., pip, conda, from source]

## Code Sample
```python
# Minimal code to reproduce the issue
from src.eeg_preprocessing import EEGPreprocessor
import numpy as np

preprocessor = EEGPreprocessor(sampling_rate=256)
data = np.random.randn(32, 2560)
result = preprocessor.preprocess(data)  # Error occurs here
```

## Data Information
- **EEG Device**: [e.g., Muse, OpenBCI, simulated data]
- **Number of Channels**: [e.g., 32]
- **Sampling Rate**: [e.g., 256 Hz]
- **Data Format**: [e.g., .edf, .mat, numpy array]
- **Data Shape**: [e.g., (32, 2560)]

## Screenshots
If applicable, add screenshots to help explain your problem.

## Additional Context
Add any other context about the problem here.

## Possible Solution (Optional)
If you have suggestions on how to fix the issue, please describe them here.

## Checklist
- [ ] I have searched existing issues to avoid duplicates
- [ ] I have provided a minimal reproducible example
- [ ] I have included my environment information
- [ ] I have checked that my dependencies are up to date
