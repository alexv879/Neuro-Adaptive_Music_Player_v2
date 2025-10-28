# Emotion Recognition → Music Recommendation: Data Flow

## Complete Pipeline

```
EEG Input (32 channels × 8064 samples)
   ↓ Preprocessing
Cleaned EEG (bandpass filter 0.5-45 Hz)
   ↓ Feature Extraction
Feature Vector (355 dimensions: band power, FAA, stats)
   ↓ Emotion Model (CNN+BiLSTM)
Emotion Prediction: 'focused', 'happy', 'sad', 'neutral', 'relaxed'
   ↓ Music Recommendation Engine
Music Track (matched to emotion profile)
```

## Emotion → Music Mapping

| Emotion | Valence | Arousal | Tempo (BPM) | Genres |
|---------|---------|---------|-------------|--------|
| HAPPY | High | High | 120-140 | pop, dance, indie |
| SAD | Low | Low | 60-100 | acoustic, indie, ambient |
| CALM | Medium | Low | 70-90 | classical, ambient, lo-fi |
| EXCITED | High | High | 130-170 | electronic, pop, rock |
| STRESSED | Low | High | 60-90 | ambient, classical, meditation |
| RELAXED | Medium | Low | 60-90 | jazz, classical, acoustic |
| **FOCUSED** | Medium | Medium | 90-110 | **lo-fi, classical, instrumental** |
| NEUTRAL | Medium | Medium | 90-120 | pop, indie, folk |
| ANGRY | Low | High | 110-140 | rock, metal, punk |

## Data Types at Each Stage

1. **EEG Input:** `numpy.ndarray(float64)` shape (32, 8064)
2. **Preprocessed:** `numpy.ndarray(float64)` shape (32, 8064)
3. **Features:** `numpy.ndarray(float64)` shape (355,)
4. **Emotion:** `str` one of 5 labels
5. **Music:** `Track` object with title, artist, genre, tempo

## Key Files

- `src/eeg_preprocessing.py` - Signal cleaning
- `src/eeg_features.py` - Feature extraction
- `src/emotion_recognition_model.py` - Emotion prediction
- `src/music_recommendation.py` - Music selection
- `src/config.py` - Configuration parameters

---
*For research basis, see RESEARCH_VALIDATION_MUSIC_MAPPING.md*
