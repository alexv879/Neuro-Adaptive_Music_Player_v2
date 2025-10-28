# Tests Directory

This directory contains the test suite for the Neuro-Adaptive Music Player v2.

## Test Structure

```
tests/
├── unit/                   # Unit tests for individual modules
│   ├── test_config.py
│   ├── test_preprocessing.py
│   ├── test_features.py
│   ├── test_model.py
│   ├── test_data_loaders.py
│   ├── test_live_eeg.py
│   ├── test_music.py
│   ├── test_personalization.py
│   └── test_utils.py
├── integration/            # Integration tests for module interactions
│   ├── test_pipeline.py
│   ├── test_real_time.py
│   └── test_training.py
├── fixtures/               # Test data and fixtures
│   ├── sample_eeg.npy
│   ├── sample_features.npy
│   ├── sample_model.h5
│   └── config_test.py
├── conftest.py             # Pytest configuration and shared fixtures
└── README.md               # This file
```

## Running Tests

### Run all tests
```bash
pytest tests/
```

### Run specific test file
```bash
pytest tests/unit/test_preprocessing.py
```

### Run with coverage
```bash
pytest tests/ --cov=src --cov-report=html
```

### Run with verbose output
```bash
pytest tests/ -v
```

### Run specific test function
```bash
pytest tests/unit/test_preprocessing.py::test_bandpass_filter
```

## Test Requirements

Install testing dependencies:

```bash
pip install pytest pytest-cov pytest-mock pytest-benchmark
```

## Writing Tests

### Unit Test Template

```python
"""
Unit tests for EEG preprocessing module.

Tests cover:
- Bandpass filtering
- Notch filtering
- Artifact detection
- Quality checking
"""

import pytest
import numpy as np
from src.eeg_preprocessing import EEGPreprocessor


@pytest.fixture
def sample_data():
    """Generate sample EEG data for testing."""
    np.random.seed(42)
    # 32 channels, 2560 samples (10 seconds @ 256 Hz)
    return np.random.randn(32, 2560)


@pytest.fixture
def preprocessor():
    """Create preprocessor instance with default settings."""
    return EEGPreprocessor(
        sampling_rate=256,
        lowcut=1.0,
        highcut=50.0,
        notch_freq=50.0
    )


def test_bandpass_filter(preprocessor, sample_data):
    """Test bandpass filtering preserves shape and scales amplitude."""
    filtered = preprocessor._apply_bandpass(sample_data)
    
    assert filtered.shape == sample_data.shape
    assert np.all(np.isfinite(filtered))
    assert filtered.std() < sample_data.std()  # Filtering reduces variance


def test_notch_filter(preprocessor, sample_data):
    """Test notch filter removes line noise."""
    filtered = preprocessor._apply_notch(sample_data)
    
    assert filtered.shape == sample_data.shape
    assert np.all(np.isfinite(filtered))
    
    # Check 50 Hz component is attenuated
    freqs = np.fft.fftfreq(sample_data.shape[1], 1/256)
    fft_orig = np.abs(np.fft.fft(sample_data[0]))
    fft_filt = np.abs(np.fft.fft(filtered[0]))
    
    idx_50hz = np.argmin(np.abs(freqs - 50.0))
    assert fft_filt[idx_50hz] < fft_orig[idx_50hz] * 0.5


def test_artifact_detection(preprocessor, sample_data):
    """Test artifact detection identifies abnormal samples."""
    # Add artificial artifact
    data_with_artifact = sample_data.copy()
    data_with_artifact[5, 100:200] = 500  # Voltage spike
    
    artifacts = preprocessor._detect_artifacts(data_with_artifact)
    
    assert len(artifacts) > 0
    assert any(100 <= a < 200 for a in artifacts)


def test_preprocess_pipeline(preprocessor, sample_data):
    """Test full preprocessing pipeline."""
    cleaned = preprocessor.preprocess(sample_data)
    
    assert cleaned.shape == sample_data.shape
    assert np.all(np.isfinite(cleaned))
    assert cleaned.mean() != sample_data.mean()  # Data was modified


def test_batch_processing(preprocessor):
    """Test preprocessing multiple trials efficiently."""
    batch = np.random.randn(10, 32, 2560)  # 10 trials
    
    cleaned_batch = preprocessor.preprocess_batch(batch)
    
    assert cleaned_batch.shape == batch.shape
    assert np.all(np.isfinite(cleaned_batch))


def test_streaming_mode(preprocessor):
    """Test streaming mode with overlapping windows."""
    # Simulate 3 consecutive windows
    windows = [np.random.randn(32, 256) for _ in range(3)]
    
    results = []
    for window in windows:
        cleaned = preprocessor.preprocess_streaming(window)
        results.append(cleaned)
    
    assert len(results) == 3
    assert all(r.shape == (32, 256) for r in results)


def test_quality_check(preprocessor, sample_data):
    """Test quality assessment functionality."""
    report = preprocessor.check_quality(sample_data)
    
    assert 'n_noisy_channels' in report
    assert 'n_artifacts' in report
    assert 'overall_quality' in report
    assert 0 <= report['overall_quality'] <= 1


def test_edge_cases(preprocessor):
    """Test edge cases and error handling."""
    # Empty array
    with pytest.raises(ValueError):
        preprocessor.preprocess(np.array([]))
    
    # Wrong shape
    with pytest.raises(ValueError):
        preprocessor.preprocess(np.random.randn(100))  # 1D instead of 2D
    
    # NaN values
    data_with_nan = np.random.randn(32, 2560)
    data_with_nan[0, 0] = np.nan
    
    with pytest.raises(ValueError):
        preprocessor.preprocess(data_with_nan)


@pytest.mark.benchmark
def test_preprocessing_performance(preprocessor, sample_data, benchmark):
    """Benchmark preprocessing speed."""
    result = benchmark(preprocessor.preprocess, sample_data)
    assert result.shape == sample_data.shape
```

### Integration Test Template

```python
"""
Integration tests for full emotion recognition pipeline.

Tests cover:
- Data loading → Preprocessing → Feature extraction → Model prediction
- Real-time processing loop
- Training workflow
"""

import pytest
import numpy as np
from src.config import Config
from src.eeg_preprocessing import EEGPreprocessor
from src.eeg_features import EEGFeatureExtractor
from src.emotion_recognition_model import EmotionRecognitionModel


@pytest.fixture
def full_pipeline():
    """Create complete pipeline components."""
    config = Config()
    
    preprocessor = EEGPreprocessor(
        sampling_rate=config.SAMPLING_RATE,
        lowcut=config.BANDPASS_LOWCUT,
        highcut=config.BANDPASS_HIGHCUT
    )
    
    feature_extractor = EEGFeatureExtractor(
        sampling_rate=config.SAMPLING_RATE,
        channel_names=config.CHANNEL_NAMES
    )
    
    model = EmotionRecognitionModel(
        input_shape=(167,),
        n_classes=5
    )
    model.build_model('dense')  # Use simple model for testing
    
    return config, preprocessor, feature_extractor, model


def test_end_to_end_prediction(full_pipeline):
    """Test complete pipeline from raw data to emotion prediction."""
    config, preprocessor, feature_extractor, model = full_pipeline
    
    # 1. Generate raw data
    raw_data = np.random.randn(32, 2560)
    
    # 2. Preprocess
    cleaned_data = preprocessor.preprocess(raw_data)
    
    # 3. Extract features
    features = feature_extractor.extract_features(cleaned_data)
    
    # 4. Predict (model untrained, just test inference)
    prediction = model.predict(features.reshape(1, -1))
    
    assert prediction.shape == (1,)
    assert 0 <= prediction[0] < 5


def test_batch_pipeline(full_pipeline):
    """Test pipeline handles batches efficiently."""
    config, preprocessor, feature_extractor, model = full_pipeline
    
    batch_size = 10
    batch_data = np.random.randn(batch_size, 32, 2560)
    
    # Process batch
    cleaned_batch = preprocessor.preprocess_batch(batch_data)
    features_batch = feature_extractor.extract_features_batch(cleaned_batch)
    predictions = model.predict(features_batch)
    
    assert predictions.shape == (batch_size,)


def test_training_workflow(full_pipeline):
    """Test model training workflow."""
    config, preprocessor, feature_extractor, model = full_pipeline
    
    # Generate synthetic training data
    X_train = np.random.randn(100, 167)
    y_train = np.random.randint(0, 5, 100)
    
    # Train for 2 epochs (quick test)
    history = model.train(
        X_train, y_train,
        epochs=2,
        batch_size=16,
        validation_split=0.2,
        verbose=0
    )
    
    assert 'loss' in history.history
    assert 'val_loss' in history.history
    assert len(history.history['loss']) == 2
```

## Test Coverage Goals

- **Overall**: >80%
- **Critical modules** (preprocessing, features, model): >90%
- **Utility functions**: >70%

Check current coverage:

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

## Continuous Integration

Tests run automatically on:
- Every push to main branch
- Every pull request
- Nightly builds

CI configuration: `.github/workflows/tests.yml`

## Test Data

Fixtures in `tests/fixtures/`:
- `sample_eeg.npy` - 10-second EEG (32 channels, 2560 samples)
- `sample_features.npy` - Pre-computed feature vector (167 dims)
- `sample_model.h5` - Small trained model for testing

Generate fixtures:

```bash
python tests/fixtures/generate_fixtures.py
```

## Benchmarking

Benchmark performance-critical functions:

```bash
pytest tests/unit/test_preprocessing.py -v --benchmark-only
```

Target performance:
- Preprocessing: <20ms per trial
- Feature extraction: <25ms per trial
- Model inference: <10ms per trial

## Debugging Tests

Run with debugger:

```bash
pytest tests/ --pdb
```

Show print statements:

```bash
pytest tests/ -s
```

Run only failed tests from last run:

```bash
pytest tests/ --lf
```

## Mocking

Use `pytest-mock` for mocking external dependencies:

```python
def test_live_eeg_connection(mocker):
    """Test EEG device connection with mock."""
    mock_serial = mocker.patch('serial.Serial')
    mock_serial.return_value.read.return_value = b'\x00' * 32
    
    handler = LiveEEGHandler(port='COM3')
    data = handler.read_chunk()
    
    assert data is not None
    mock_serial.assert_called_once()
```

## Parametrized Tests

Test multiple configurations efficiently:

```python
@pytest.mark.parametrize("architecture", ['dense', 'cnn', 'bilstm', 'cnn_bilstm'])
def test_all_architectures(architecture):
    """Test all model architectures."""
    model = EmotionRecognitionModel(input_shape=(167,), n_classes=5)
    model.build_model(architecture)
    
    X_dummy = np.random.randn(10, 167)
    predictions = model.predict(X_dummy)
    
    assert predictions.shape == (10,)
```

## Test Fixtures

Shared fixtures in `conftest.py`:

```python
@pytest.fixture(scope='session')
def sample_dataset():
    """Load sample dataset once for all tests."""
    X = np.random.randn(100, 167)
    y = np.random.randint(0, 5, 100)
    return X, y


@pytest.fixture
def temp_model_path(tmp_path):
    """Provide temporary path for model saving."""
    return tmp_path / "test_model.h5"
```

## Contributing Tests

When adding new features:
1. Write tests FIRST (TDD approach)
2. Ensure >80% coverage for new code
3. Add integration tests for interactions
4. Update this README if adding test categories
5. Run full test suite before committing

## CI/CD Pipeline

GitHub Actions workflow:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    - uses: actions/setup-python@v2
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: pytest tests/ --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

## License

Tests are part of the main project and use the same license (see LICENSE).
