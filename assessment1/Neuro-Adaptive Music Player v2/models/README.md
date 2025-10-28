# Models Directory

This directory stores trained emotion recognition models.

## Structure

```
models/
├── checkpoints/        # Training checkpoints
├── pretrained/         # Pre-trained models
├── personalized/       # User-specific fine-tuned models
├── experiments/        # Experimental model variants
└── README.md           # This file
```

## Model Files

Models are saved in HDF5 format (.h5) with accompanying metadata:
- `model_name.h5` - Keras model architecture + weights
- `model_name_encoder.pkl` - Label encoder for emotion classes
- `model_name_config.json` - Training configuration
- `model_name_history.pkl` - Training history

## Pre-trained Models

Pre-trained models will be available for download:

### DEAP-trained Model
- **Accuracy**: ~78% (5-class emotion)
- **Architecture**: CNN+BiLSTM hybrid
- **Features**: 167-dimensional (32 channels × 5 bands + 3 FAA + stats)
- **Download**: [Link to be added]

### SEED-trained Model
- **Accuracy**: ~85% (3-class emotion)
- **Architecture**: CNN+BiLSTM hybrid
- **Features**: 310-dimensional (62 channels × 5 bands)
- **Download**: [Link to be added]

## Using Pre-trained Models

```python
from src.emotion_recognition_model import EmotionRecognitionModel

# Load pre-trained model
model = EmotionRecognitionModel(input_shape=(167,))
model.load_model('models/pretrained/deap_cnn_bilstm.h5')

# Predict
emotion = model.predict(features)
print(f"Detected emotion: {emotion[0]}")
```

## Training Your Own Model

```python
from src.emotion_recognition_model import EmotionRecognitionModel
import numpy as np

# Prepare data
X_train = np.load('data/processed/X_train.npy')
y_train = np.load('data/processed/y_train.npy')

# Create and train model
model = EmotionRecognitionModel(
    input_shape=(X_train.shape[1],),
    n_classes=5,
    model_name='my_custom_model'
)

model.build_model(architecture='cnn_bilstm')

# Train with checkpointing
model.train(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2
)

# Save trained model
model.save_model('models/my_custom_model.h5')
```

## Model Architectures

Available architectures:

### 1. CNN+BiLSTM (Recommended)
- **Best for**: General emotion recognition
- **Pros**: Captures spatial and temporal patterns
- **Cons**: Slower inference (~10ms)
- **Accuracy**: 75-90% on standard datasets

### 2. CNN-only
- **Best for**: Real-time applications
- **Pros**: Fast inference (~5ms)
- **Cons**: Misses temporal dependencies
- **Accuracy**: 70-85%

### 3. BiLSTM-only
- **Best for**: Temporal sequence modeling
- **Pros**: Good for time-series
- **Cons**: Slow training
- **Accuracy**: 65-80%

### 4. Dense (MLP)
- **Best for**: Baseline comparison
- **Pros**: Simple, interpretable
- **Cons**: Lower accuracy
- **Accuracy**: 60-75%

## Fine-tuning (Transfer Learning)

To adapt a pre-trained model to your data:

```python
from src.model_personalization import fine_tune_model

# Load pre-trained model
base_model = EmotionRecognitionModel(input_shape=(167,))
base_model.load_model('models/pretrained/deap_cnn_bilstm.h5')

# Fine-tune on your data (requires src/model_personalization.py)
fine_tuned = fine_tune_model(
    base_model,
    X_personal, y_personal,
    freeze_layers=10,
    epochs=20
)

# Save personalized model
fine_tuned.save_model('models/personalized/my_model.h5')
```

## Model Performance

Track model performance with these metrics:

```python
# Evaluate on test set
results = model.evaluate(X_test, y_test)

print(f"Accuracy: {results['accuracy']:.4f}")
print("\nClassification Report:")
print(results['classification_report'])
print("\nConfusion Matrix:")
print(results['confusion_matrix'])
```

## Checkpointing

Models are automatically checkpointed during training:
- Saved to `models/checkpoints/`
- Only best model (highest validation accuracy) is kept
- Resume training from checkpoint if interrupted

## Model Versioning

Version your models with timestamps:

```python
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = f"emotion_cnn_bilstm_{timestamp}"

model = EmotionRecognitionModel(
    input_shape=(167,),
    model_name=model_name
)
```

## .gitignore

Model files are excluded from git by default (too large):
- `*.h5`
- `*.pkl`
- `*.pb`
- `*.ckpt`

Use Git LFS or external storage (Google Drive, Dropbox) for sharing models.

## Optimization

### For Faster Inference
1. Use CNN-only architecture
2. Reduce number of filters
3. Use smaller dense layers
4. Quantize model (TensorFlow Lite)

### For Better Accuracy
1. Use CNN+BiLSTM architecture
2. Increase model capacity
3. Add more training data
4. Use data augmentation
5. Ensemble multiple models

## Hardware Requirements

### Training
- **CPU**: Possible but slow (hours)
- **GPU**: Recommended (NVIDIA with CUDA)
  - Training time: ~10-30 minutes for 100 epochs
  - Memory: 4GB+ VRAM

### Inference
- **CPU**: Sufficient for most applications
- **GPU**: Optional (3x faster)
- **Edge devices**: Possible with quantization

## Citation

If you use our pre-trained models in research:

```bibtex
@software{neuro_adaptive_music_player_v2,
  author = {Alexander V.},
  title = {Neuro-Adaptive Music Player v2: Pre-trained EEG Emotion Recognition Models},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/alexv879/neuro-adaptive-music-player-v2}
}
```
