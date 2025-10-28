# Emotion Recognition → Music Recommendation: Data Flow

**Date:** October 28, 2025  
**Purpose:** Document how emotion predictions are passed to music recommendation

---

## 🔄 COMPLETE DATA FLOW

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         1. EEG DATA INPUT                                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    Raw EEG signals (32 channels × 8064 samples)
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    2. PREPROCESSING (eeg_preprocessing.py)               │
│  - Bandpass filter (0.5-45 Hz)                                          │
│  - Artifact removal                                                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    Cleaned EEG signals (32 × 8064)
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                 3. FEATURE EXTRACTION (eeg_features.py)                  │
│  - Band power (delta, theta, alpha, beta, gamma)                        │
│  - Frontal Alpha Asymmetry (FAA)                                        │
│  - Statistical features                                                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    Feature vector (355 dimensions)
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              4. EMOTION RECOGNITION (emotion_recognition_model.py)       │
│  - CNN+BiLSTM deep learning model                                       │
│  - Input: (355,) feature vector                                         │
│  - Output: Emotion predictions                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                          ╔═══════════════════╗
                          ║  EMOTION OUTPUT   ║
                          ╚═══════════════════╝
                                    │
            ┌───────────────────────┼───────────────────────┐
            ▼                       ▼                       ▼
    ┌──────────────┐        ┌──────────────┐      ┌──────────────┐
    │   Valence    │        │   Arousal    │      │   Emotion    │
    │  (2 classes) │        │  (2 classes) │      │  (5 classes) │
    │              │        │              │      │              │
    │ [0.2, 0.8]   │        │ [0.7, 0.3]   │      │[0.05, 0.82,  │
    │  ↓     ↓     │        │  ↓     ↓     │      │ 0.03, 0.07,  │
    │ Neg   Pos    │        │ Low  High    │      │ 0.03]        │
    │              │        │              │      │  ↓           │
    │              │        │              │      │ happy (82%)  │
    └──────────────┘        └──────────────┘      └──────────────┘
                                    │
                                    ▼
                    **PRIMARY OUTPUT: EMOTION LABEL**
                        String: "happy" or "sad" or "relaxed"
                                  or "focused" or "neutral"
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│           5. MUSIC RECOMMENDATION (music_recommendation.py)              │
│  - Input: emotion (string), confidence (float)                          │
│  - Emotion-to-genre mapping                                             │
│  - Spotify/YouTube search                                               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                          ╔═══════════════════╗
                          ║   MUSIC TRACK     ║
                          ╚═══════════════════╝
                                    │
                                    ▼
                    Track(title="Happy", artist="Pharrell",
                          uri="spotify:track:60nZcImuf...", ...)
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        6. PLAYBACK CONTROL                               │
│  - Start/pause/skip music                                               │
│  - Log user feedback                                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 DETAILED OUTPUT SPECIFICATIONS

### 1. Emotion Recognition Model Output

**Method:** `model.predict(X_test)` in `emotion_recognition_model.py`

#### For Hierarchical Model (cnn_bilstm):
```python
predictions = model.model.predict(X_test, verbose=0)
# Returns: List[np.ndarray, np.ndarray, np.ndarray]

# predictions[0] = Valence probabilities (n_samples, 2)
[[0.2, 0.8],   # Sample 1: 80% positive
 [0.6, 0.4],   # Sample 2: 60% negative
 [0.1, 0.9]]   # Sample 3: 90% positive

# predictions[1] = Arousal probabilities (n_samples, 2)
[[0.7, 0.3],   # Sample 1: 30% high arousal
 [0.4, 0.6],   # Sample 2: 60% high arousal
 [0.8, 0.2]]   # Sample 3: 20% high arousal

# predictions[2] = Emotion probabilities (n_samples, 5)
[[0.05, 0.82, 0.03, 0.07, 0.03],  # Sample 1: 82% happy
 [0.10, 0.15, 0.60, 0.10, 0.05],  # Sample 2: 60% sad
 [0.02, 0.85, 0.05, 0.06, 0.02]]  # Sample 3: 85% happy
```

#### After Processing (convert to labels):
```python
emotion_indices = np.argmax(predictions[2], axis=1)  # [1, 2, 1]
emotion_labels = model.label_encoder.inverse_transform(emotion_indices)
# Returns: array(['happy', 'sad', 'happy'], dtype=object)
```

**Output Type:** `numpy.ndarray` of **strings**
**Example Values:** `'happy'`, `'sad'`, `'relaxed'`, `'focused'`, `'neutral'`

---

### 2. Music Recommendation Engine Input

**Method:** `engine.recommend(emotion, confidence)` in `music_recommendation.py`

#### Expected Input:
```python
def recommend(
    self,
    emotion: Union[str, EmotionCategory],  # ← Can be STRING or enum
    confidence: float = 1.0,                # ← Optional confidence score
    n_tracks: int = 1,
    diversity: float = 0.3
) -> Union[Track, List[Track]]:
```

#### Input Processing (line 329-335):
```python
# Converts string to EmotionCategory enum
if isinstance(emotion, str):
    try:
        emotion = EmotionCategory(emotion.lower())  # 'happy' → EmotionCategory.HAPPY
    except ValueError:
        logger.warning(f"Unknown emotion '{emotion}', defaulting to NEUTRAL")
        emotion = EmotionCategory.NEUTRAL
```

#### Supported Emotion Strings:
```python
class EmotionCategory(Enum):
    CALM = "calm"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    NEUTRAL = "neutral"
    EXCITED = "excited"
    RELAXED = "relaxed"
    STRESSED = "stressed"
```

---

## 🔗 CONNECTING THE TWO MODULES

### Current State: **MISMATCH EXISTS**

#### Emotion Recognition Model Outputs (5 classes):
```python
# From config.py (line 139)
EMOTION_LABELS: Dict[int, str] = {
    0: 'neutral',
    1: 'happy',
    2: 'sad',
    3: 'relaxed',
    4: 'focused',
}
```

#### Music Recommendation Engine Expects (8 classes):
```python
# From music_recommendation.py (line 47)
class EmotionCategory(Enum):
    CALM = "calm"           # ✗ NOT in model output
    HAPPY = "happy"         # ✓ MATCH
    SAD = "sad"             # ✓ MATCH
    ANGRY = "angry"         # ✗ NOT in model output
    NEUTRAL = "neutral"     # ✓ MATCH
    EXCITED = "excited"     # ✗ NOT in model output
    RELAXED = "relaxed"     # ✓ MATCH
    STRESSED = "stressed"   # ✗ NOT in model output
```

### ⚠️ MAPPING REQUIRED

You need an **emotion mapping layer** between the two modules:

```python
# Add to your main application
def map_emotion_to_music_category(emotion: str) -> str:
    """
    Map emotion recognition output to music recommendation input.
    
    Args:
        emotion: Output from emotion_recognition_model ('happy', 'sad', etc.)
        
    Returns:
        Compatible emotion for music_recommendation ('happy', 'calm', etc.)
    """
    # Direct matches (no mapping needed)
    direct_matches = {'happy', 'sad', 'neutral', 'relaxed'}
    if emotion in direct_matches:
        return emotion
    
    # Map 'focused' → closest music category
    if emotion == 'focused':
        return 'calm'  # or 'relaxed' depending on your preference
    
    # Fallback
    return 'neutral'
```

---

## 💡 EXAMPLE: COMPLETE FLOW

### Code Example
```python
import numpy as np
from src.emotion_recognition_model import EmotionRecognitionModel
from src.music_recommendation import MusicRecommendationEngine, MusicPlatform

# 1. Load trained emotion recognition model
model = EmotionRecognitionModel(input_shape=(355,), n_classes=5)
model.load_model('models/emotion_model.h5')

# 2. Extract features from EEG (assume you have this)
X_features = extract_features_from_eeg(eeg_signal)  # Shape: (1, 355)

# 3. Predict emotion
emotion_label = model.predict(X_features)  # Returns: array(['happy'])
emotion_str = emotion_label[0]  # Convert to string: 'happy'

# 4. Get confidence (optional)
emotion_probs = model.predict_proba(X_features)  # Shape: (1, 5)
confidence = np.max(emotion_probs)  # e.g., 0.82

# 5. Map emotion (if needed)
def map_emotion(emo):
    return 'calm' if emo == 'focused' else emo

music_emotion = map_emotion(emotion_str)  # 'happy' stays 'happy'

# 6. Get music recommendation
engine = MusicRecommendationEngine(platform=MusicPlatform.SPOTIFY)
engine.authenticate_spotify_simple()  # Setup Spotify

track = engine.recommend(
    emotion=music_emotion,     # 'happy'
    confidence=confidence,      # 0.82
    n_tracks=1
)

# 7. Play the track
engine.play(track)

print(f"Detected emotion: {emotion_str} (confidence: {confidence:.2%})")
print(f"Playing: {track.artist} - {track.title}")
```

### Output:
```
Detected emotion: happy (confidence: 82%)
Playing: Pharrell Williams - Happy
```

---

## 📋 DATA TYPE SPECIFICATIONS

### Emotion Recognition Model → Output

| Type | Format | Example | Purpose |
|------|--------|---------|---------|
| **Primary Output** | `np.ndarray[str]` | `array(['happy', 'sad'])` | Main emotion labels |
| **Probabilities** | `np.ndarray[float]` | `array([[0.05, 0.82, ...]])` | Confidence scores |
| **Shape** | `(n_samples,)` | `(40,)` for 40 trials | One label per sample |

**Code to extract:**
```python
# Get labels
emotions = model.predict(X_test)  # array(['happy', 'sad', ...])

# Get probabilities
probs = model.predict_proba(X_test)  # (n_samples, 5)
confidences = np.max(probs, axis=1)  # (n_samples,)
```

### Music Recommendation Engine → Input

| Parameter | Type | Required | Default | Example |
|-----------|------|----------|---------|---------|
| `emotion` | `str` or `EmotionCategory` | ✅ Yes | - | `'happy'` or `EmotionCategory.HAPPY` |
| `confidence` | `float` | ❌ No | `1.0` | `0.82` |
| `n_tracks` | `int` | ❌ No | `1` | `3` |
| `diversity` | `float` | ❌ No | `0.3` | `0.5` |

**Code to call:**
```python
# Simple (just emotion)
track = engine.recommend('happy')

# With confidence
track = engine.recommend('happy', confidence=0.82)

# Multiple tracks
tracks = engine.recommend('happy', confidence=0.82, n_tracks=5)
```

### Music Recommendation Engine → Output

| Type | Format | Example |
|------|--------|---------|
| **Single Track** | `Track` object | `Track(title="Happy", artist="Pharrell", ...)` |
| **Multiple Tracks** | `List[Track]` | `[Track(...), Track(...), Track(...)]` |

**Track Object Structure:**
```python
@dataclass
class Track:
    title: str              # "Happy"
    artist: str             # "Pharrell Williams"
    uri: str                # "spotify:track:60nZcImuf6D9Gn9N7SgAJ"
    platform: MusicPlatform # MusicPlatform.SPOTIFY
    emotion: EmotionCategory # EmotionCategory.HAPPY
    duration_ms: int        # 233320 (3:53)
    album: str              # "G I R L"
    popularity: int         # 87
    energy: float           # 0.78
    valence: float          # 0.92
    tempo: float            # 160.0
```

**Access track info:**
```python
track = engine.recommend('happy')
print(track.title)    # "Happy"
print(track.artist)   # "Pharrell Williams"
print(str(track))     # "Pharrell Williams - Happy"
engine.play(track)    # Plays on Spotify
```

---

## 🎯 INTEGRATION CHECKLIST

### ✅ What Works Now:
- ✅ Model outputs string labels: `'happy'`, `'sad'`, `'neutral'`, `'relaxed'`, `'focused'`
- ✅ Music engine accepts string input: `engine.recommend('happy')`
- ✅ Direct matches work: `'happy'` → `'happy'`, `'sad'` → `'sad'`, `'neutral'` → `'neutral'`, `'relaxed'` → `'relaxed'`

### ⚠️ What Needs Fixing:
- ⚠️ **`'focused'` has no music profile** - Need to map to `'calm'` or `'relaxed'`
- ⚠️ **Case sensitivity** - Model outputs lowercase, engine converts to lowercase (OK)
- ⚠️ **Confidence not used** - Music engine accepts it but doesn't do anything special with it yet

### 🔧 Recommended Solution:

**Create integration module:** `emotion_music_bridge.py`

```python
"""
Bridge between emotion recognition and music recommendation.
Handles emotion mapping and confidence processing.
"""

from typing import Tuple
import numpy as np
from src.emotion_recognition_model import EmotionRecognitionModel
from src.music_recommendation import MusicRecommendationEngine, Track

class EmotionMusicBridge:
    """Bridge between emotion detection and music recommendation."""
    
    # Emotion mapping (model output → music input)
    EMOTION_MAP = {
        'happy': 'happy',
        'sad': 'sad',
        'neutral': 'neutral',
        'relaxed': 'relaxed',
        'focused': 'calm',  # Map focused to calm music
    }
    
    @staticmethod
    def predict_and_recommend(
        model: EmotionRecognitionModel,
        music_engine: MusicRecommendationEngine,
        eeg_features: np.ndarray
    ) -> Tuple[str, float, Track]:
        """
        Complete pipeline: predict emotion → recommend music.
        
        Args:
            model: Trained emotion recognition model
            music_engine: Music recommendation engine
            eeg_features: Extracted EEG features (n_samples, n_features)
            
        Returns:
            (emotion, confidence, track)
        """
        # 1. Predict emotion
        emotion = model.predict(eeg_features)[0]  # Get first prediction
        
        # 2. Get confidence
        probs = model.predict_proba(eeg_features)
        confidence = float(np.max(probs[0]))
        
        # 3. Map emotion to music category
        music_emotion = EmotionMusicBridge.EMOTION_MAP.get(
            emotion, 
            'neutral'  # Fallback
        )
        
        # 4. Recommend music
        track = music_engine.recommend(
            emotion=music_emotion,
            confidence=confidence,
            n_tracks=1
        )
        
        return emotion, confidence, track

# Usage:
# bridge = EmotionMusicBridge()
# emotion, conf, track = bridge.predict_and_recommend(model, engine, features)
```

---

## 📝 SUMMARY

### Output Format Chain:

```
EEG Signal
    ↓ (preprocessing)
Feature Vector (355 dimensions)
    ↓ (model.predict)
Emotion Label: "happy" (string)
    ↓ (optional mapping)
Music Emotion: "happy" (string)
    ↓ (engine.recommend)
Track Object (title, artist, uri, ...)
    ↓ (engine.play)
Music Playing on Spotify
```

### Key Points:
1. **Model output:** String label (`'happy'`, `'sad'`, etc.)
2. **Music input:** Same string label (or EmotionCategory enum)
3. **Connection:** Pass string directly (with optional mapping for `'focused'`)
4. **Confidence:** Optional but available from `predict_proba()`
5. **Track:** Rich object with all metadata for playback

### Quick Integration:
```python
# Simple 3-line integration
emotion = model.predict(features)[0]          # "happy"
track = music_engine.recommend(emotion)       # Track object
success = music_engine.play(track)            # True/False
```

**That's it!** The interface is actually very clean - just pass the string emotion label from the model directly to the music engine. 🎵
