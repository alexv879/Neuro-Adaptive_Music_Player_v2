# Examples Directory

This directory contains example scripts and tutorials for using the Neuro-Adaptive Music Player v2.

## Available Examples

### Completed Examples
None yet - examples will be added as modules are completed.

### Planned Examples

1. **`01_basic_preprocessing.py`**
   - Load raw EEG data
   - Apply bandpass and notch filtering
   - Detect artifacts
   - Visualize cleaned signals

2. **`02_feature_extraction.py`**
   - Extract frequency band powers
   - Calculate frontal alpha asymmetry
   - Compute statistical features
   - Save feature vectors

3. **`03_train_emotion_model.py`**
   - Load DEAP or SEED dataset
   - Extract features from all trials
   - Train CNN+BiLSTM model
   - Evaluate performance

4. **`04_real_time_detection.py`**
   - Connect to EEG device (Muse, OpenBCI)
   - Process streaming data
   - Predict emotions in real-time
   - Display results

5. **`05_music_recommendation.py`**
   - Detect current emotion
   - Query music database
   - Recommend mood-matching music
   - Play songs

6. **`06_model_personalization.py`**
   - Load pre-trained model
   - Fine-tune on personal data
   - Compare baseline vs personalized
   - Save personalized model

7. **`07_batch_processing.py`**
   - Process multiple files
   - Extract features efficiently
   - Save to HDF5 format
   - Performance benchmarking

8. **`08_advanced_visualization.py`**
   - Plot EEG topomaps
   - Visualize frequency bands
   - Display emotion predictions
   - Real-time dashboards

9. **`09_experiment_comparison.py`**
   - Compare different architectures
   - Test hyperparameter variations
   - Statistical significance tests
   - Generate report

10. **`10_deployment_demo.py`**
    - Load optimized model
    - Minimize latency
    - Production-ready inference
    - Error handling

## Usage

Once examples are created, run them with:

```bash
python examples/01_basic_preprocessing.py
```

Or from Python:

```python
import sys
sys.path.append('.')
from examples import basic_preprocessing

basic_preprocessing.main()
```

## Requirements

Examples may require additional dependencies:

```bash
pip install matplotlib seaborn plotly dash
pip install pygame spotipy  # For music examples
pip install pyserial bleak  # For hardware examples
```

## Directory Structure

```
examples/
├── data/                    # Example datasets (small samples)
│   ├── sample_eeg.edf       # 10-second EEG recording
│   ├── sample_features.npy  # Pre-computed features
│   └── README.md
├── outputs/                 # Example outputs
│   ├── plots/               # Generated visualizations
│   ├── models/              # Trained models from examples
│   └── logs/                # Execution logs
├── notebooks/               # Jupyter notebooks (alternative format)
│   ├── 01_preprocessing_tutorial.ipynb
│   ├── 02_feature_extraction_tutorial.ipynb
│   └── ...
└── README.md                # This file
```

## Contributing Examples

When creating examples:

1. **Keep it simple**: Focus on one concept per example
2. **Add comments**: Explain what each section does
3. **Handle errors**: Use try-except for robustness
4. **Provide data**: Include small sample datasets
5. **Document outputs**: Show expected results
6. **Test thoroughly**: Ensure examples run on fresh install

Example template:

```python
"""
Example 01: Basic EEG Preprocessing

This script demonstrates:
- Loading raw EEG data from EDF file
- Applying bandpass filtering
- Detecting artifacts
- Saving cleaned data

Requirements:
    - pyedflib
    - numpy
    - scipy

Usage:
    python examples/01_basic_preprocessing.py
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import Config
from src.eeg_preprocessing import EEGPreprocessor


def main():
    """Main example function."""
    print("=== EEG Preprocessing Example ===\n")
    
    # 1. Setup configuration
    config = Config()
    config.validate()
    print(f"✓ Configuration loaded\n")
    
    # 2. Initialize preprocessor
    preprocessor = EEGPreprocessor(
        sampling_rate=config.SAMPLING_RATE,
        lowcut=config.BANDPASS_LOWCUT,
        highcut=config.BANDPASS_HIGHCUT,
        notch_freq=config.NOTCH_FREQ
    )
    print(f"✓ Preprocessor initialized\n")
    
    # 3. Load data
    data_path = Path("examples/data/sample_eeg.edf")
    if not data_path.exists():
        print(f"❌ Sample data not found: {data_path}")
        print("Please download sample data or use your own EEG file.")
        return
    
    # Load your data here (example with numpy)
    raw_data = np.random.randn(32, 2560)  # 32 channels, 10 seconds @ 256 Hz
    print(f"✓ Loaded data: {raw_data.shape}\n")
    
    # 4. Apply preprocessing
    try:
        cleaned_data = preprocessor.preprocess(raw_data)
        print(f"✓ Preprocessing complete")
        print(f"  - Shape: {cleaned_data.shape}")
        print(f"  - Mean: {cleaned_data.mean():.4f}")
        print(f"  - Std: {cleaned_data.std():.4f}\n")
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        return
    
    # 5. Check quality
    quality_report = preprocessor.check_quality(cleaned_data)
    print(f"✓ Quality check:")
    print(f"  - Noisy channels: {quality_report['n_noisy_channels']}")
    print(f"  - Artifacts detected: {quality_report['n_artifacts']}")
    print(f"  - Overall quality: {quality_report['overall_quality']:.2%}\n")
    
    # 6. Save results
    output_path = Path("examples/outputs/cleaned_data.npy")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, cleaned_data)
    print(f"✓ Saved cleaned data to: {output_path}")
    
    print("\n=== Example Complete! ===")


if __name__ == "__main__":
    main()
```

## Getting Help

If an example doesn't work:

1. Check you have all required dependencies
2. Verify sample data files exist
3. Read error messages carefully
4. Check the troubleshooting section in README.md
5. Open an issue on GitHub with:
   - Example name
   - Error message
   - Python version
   - OS and hardware info

## Jupyter Notebooks

Interactive notebook versions will be provided in `examples/notebooks/`. These include:
- Step-by-step explanations
- Visualizations inline
- Editable code cells
- Markdown documentation

Launch with:

```bash
jupyter notebook examples/notebooks/
```

## Sample Data

Small sample datasets are provided in `examples/data/`:
- `sample_eeg.edf` - 10-second EEG recording (32 channels)
- `sample_features.npy` - Pre-computed feature vector
- `sample_labels.npy` - Emotion labels for samples

These are for demonstration only. For real training, use full datasets:
- DEAP: [https://www.eecs.qmul.ac.uk/mmv/datasets/deap/](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/)
- SEED: [https://bcmi.sjtu.edu.cn/home/seed/](https://bcmi.sjtu.edu.cn/home/seed/)

## License

Examples are licensed under the same terms as the main project (see LICENSE).

## Citation

If you use these examples in research or teaching:

```bibtex
@software{neuro_adaptive_music_player_v2_examples,
  author = {Alexander V.},
  title = {Neuro-Adaptive Music Player v2: Tutorial Examples},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/alexv879/neuro-adaptive-music-player-v2}
}
```
