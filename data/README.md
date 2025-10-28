# Data Directory

This directory is for storing EEG datasets.

## Structure

```
data/
├── DEAP/          # DEAP dataset files (.mat)
├── SEED/          # SEED dataset files (.mat)
├── raw/           # Raw EEG recordings (.edf, .csv)
├── processed/     # Preprocessed data (.npy, .pkl)
└── README.md      # This file
```

## Datasets

### DEAP Database
- **Download**: http://www.eecs.qmul.ac.uk/mmv/datasets/deap/
- **Format**: MATLAB .mat files
- **Content**: 32 participants, 40 video trials each
- **Channels**: 32 EEG + 8 peripheral
- **Sampling Rate**: 128 Hz (originally 512 Hz, downsampled)
- **Labels**: Valence, arousal, dominance, liking

### SEED Database
- **Download**: http://bcmi.sjtu.edu.cn/~seed/
- **Format**: MATLAB .mat files
- **Content**: 15 participants, 15 film clips each
- **Channels**: 62 EEG channels
- **Sampling Rate**: 200 Hz
- **Labels**: Positive, neutral, negative emotions

### Custom Data
Place your own EEG recordings here:
- **EDF files**: Clinical EEG format (use pyedflib to load)
- **CSV files**: Custom format (create appropriate loader)
- **NumPy arrays**: Preprocessed data (.npy files)

## Loading Data

Example usage with data loaders (once implemented):

```python
from src.data_loaders import DEAPLoader, SEEDLoader

# Load DEAP dataset
deap_loader = DEAPLoader()
X, y, metadata = deap_loader.load('data/DEAP/s01.mat')

# Load SEED dataset
seed_loader = SEEDLoader()
X, y, metadata = seed_loader.load('data/SEED/1_20131027.mat')
```

## Data Format

All loaders should return data in standardized format:
- **X**: Shape (n_trials, n_channels, n_samples) - EEG data
- **y**: Shape (n_trials,) or (n_trials, n_labels) - Labels
- **metadata**: Dict with channel names, sampling rate, etc.

## Preprocessing

To preprocess raw data:

```python
from src.eeg_preprocessing import EEGPreprocessor
import numpy as np

preprocessor = EEGPreprocessor(fs=256)

# Load raw data
raw_data = np.load('data/raw/recording.npy')

# Preprocess
clean_data = preprocessor.preprocess(raw_data, apply_notch=True)

# Save
np.save('data/processed/recording_clean.npy', clean_data)
```

## .gitignore

Large data files are excluded from git by default:
- `*.edf`
- `*.mat`
- `*.npy`
- `*.pkl`

Only small example files or scripts should be committed.

## License and Ethics

When using public datasets:
1. Cite the original papers
2. Follow dataset-specific licenses
3. Respect participant privacy
4. Don't redistribute without permission
5. Use only for intended purposes (research/education)

## Citations

### DEAP
```bibtex
@article{koelstra2012deap,
  title={DEAP: A database for emotion analysis using physiological signals},
  author={Koelstra, Sander and others},
  journal={IEEE transactions on affective computing},
  volume={3},
  number={1},
  pages={18--31},
  year={2012}
}
```

### SEED
```bibtex
@article{zheng2015investigating,
  title={Investigating critical frequency bands and channels for EEG-based emotion recognition with deep neural networks},
  author={Zheng, Wei-Long and Lu, Bao-Liang},
  journal={IEEE Transactions on Autonomous Mental Development},
  volume={7},
  number={3},
  pages={162--175},
  year={2015}
}
```
