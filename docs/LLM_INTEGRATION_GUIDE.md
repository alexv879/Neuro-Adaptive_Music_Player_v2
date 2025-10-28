# LLM Music Recommender - Integration Guide

## Overview

The **LLM Music Recommender** is a new module for the Neuro-Adaptive Music Player v2 that uses OpenAI's GPT models to generate dynamic, context-aware music recommendations based on real-time EEG emotion detection.

## Key Features

✅ **Dynamic AI Recommendations** - Every track suggestion is generated in real-time by GPT-4/GPT-4o  
✅ **Context-Aware Prompts** - Considers time of day, activity, user preferences, and listening history  
✅ **Multiple Prompt Templates** - Basic, Detailed, Contextual, and Therapeutic modes  
✅ **Emotion-to-Mood Mapping** - Automatically converts EmotionCategory to descriptive mood tags  
✅ **Graceful Fallback** - Works without OpenAI API key using curated recommendations  
✅ **History Tracking** - Records all recommendations for analysis and personalization  
✅ **Production-Ready** - Type hints, error handling, comprehensive testing  

## Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│  EEG Data   │ ───> │  Emotion     │ ───> │  Mood Tag   │
│  (32 ch)    │      │  Detection   │      │  Generator  │
└─────────────┘      └──────────────┘      └─────────────┘
                            │                      │
                            v                      v
                     EmotionCategory        "happy and energetic"
                       (HAPPY)              (with confidence: 85%)
                                                   │
                                                   v
┌───────────────────────────────────────────────────────────────┐
│                    LLM Music Recommender                       │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  Prompt Builder                                          │ │
│  │  • Mood tag: "happy and energetic"                       │ │
│  │  • Context: Morning, studying, prefers instrumental      │ │
│  │  • History: Recent tracks to avoid                       │ │
│  │  • Template: Contextual (therapeutic, detailed, basic)   │ │
│  └──────────────────────────────────────────────────────────┘ │
│                           │                                    │
│                           v                                    │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  OpenAI GPT-4o API                                       │ │
│  │  Dynamic prompt → Creative recommendations               │ │
│  └──────────────────────────────────────────────────────────┘ │
│                           │                                    │
│                           v                                    │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  Response Parser                                         │ │
│  │  Extracts: Artist, Title, Reasoning                      │ │
│  └──────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────┘
                            │
                            v
              ┌─────────────────────────────┐
              │  Track Recommendations       │
              │  1. Pharrell - Happy         │
              │  2. Daft Punk - Get Lucky    │
              │  3. Mark Ronson - Uptown Funk│
              └─────────────────────────────┘
                            │
                            v
              ┌─────────────────────────────┐
              │  Spotify/YouTube Playback    │
              └─────────────────────────────┘
```

## Installation

### 1. Install OpenAI Python Library

```bash
pip install openai>=1.0.0
```

### 2. Set API Key

**Option A: Environment Variable (Recommended)**
```bash
# Windows PowerShell
$env:OPENAI_API_KEY="sk-your-api-key-here"

# Linux/Mac
export OPENAI_API_KEY="sk-your-api-key-here"
```

**Option B: Direct in Code**
```python
from src.llm_music_recommender import LLMMusicRecommender

recommender = LLMMusicRecommender(api_key="sk-your-api-key-here")
```

**Option C: Config File**
```python
# Add to config.py
OPENAI_API_KEY = "sk-your-api-key-here"
```

### 3. Verify Installation

```bash
cd "Neuro-Adaptive Music Player v2"
python tests/test_llm_recommender.py
```

## Usage

### Basic Usage

```python
from src.llm_music_recommender import LLMMusicRecommender

# Initialize recommender
recommender = LLMMusicRecommender(
    api_key="sk-...",  # Or set OPENAI_API_KEY env var
    model="gpt-4o",    # or "gpt-4", "gpt-3.5-turbo"
    temperature=0.7     # Creativity level (0-2)
)

# Get recommendations
tracks = recommender.recommend(
    mood_tag="happy and energetic",
    confidence=0.85,
    n_tracks=3
)

# Display results
for i, track in enumerate(tracks, 1):
    print(f"{i}. {track.artist} - {track.title}")
    if track.reasoning:
        print(f"   → {track.reasoning}")
```

### Integration with Emotion Detection

```python
from src import (
    Config,
    EEGPreprocessor,
    EEGFeatureExtractor,
    EmotionRecognitionModel,
    LLMMusicRecommender,
    emotion_to_mood_tag
)

# Setup pipeline
config = Config()
preprocessor = EEGPreprocessor(fs=config.SAMPLING_RATE)
feature_extractor = EEGFeatureExtractor(fs=config.SAMPLING_RATE)
emotion_model = EmotionRecognitionModel(input_shape=(167,), n_classes=5)
llm_recommender = LLMMusicRecommender(model="gpt-4o")

# Process EEG and get recommendations
def process_eeg_and_recommend(eeg_data):
    # 1. Preprocess
    clean_eeg = preprocessor.preprocess(eeg_data)
    
    # 2. Extract features
    features_dict = feature_extractor.extract_all_features(clean_eeg)
    features = feature_extractor.features_to_vector(features_dict)
    
    # 3. Detect emotion
    emotion_probs = emotion_model.predict_proba(features.reshape(1, -1))[0]
    emotion_idx = emotion_probs.argmax()
    confidence = emotion_probs[emotion_idx]
    emotion = EmotionCategory(emotion_idx)
    
    # 4. Generate mood tag
    mood_tag = emotion_to_mood_tag(emotion, confidence)
    
    # 5. Get LLM recommendations
    tracks = llm_recommender.recommend(
        mood_tag=mood_tag,
        confidence=confidence,
        n_tracks=3,
        extra_context={
            "time_of_day": "afternoon",
            "activity": "studying"
        }
    )
    
    return emotion, tracks

# Use it
emotion, recommended_tracks = process_eeg_and_recommend(raw_eeg_data)
print(f"Detected: {emotion.value}")
print(f"Recommendations: {[str(t) for t in recommended_tracks]}")
```

### Context-Aware Recommendations

```python
# Enrich recommendations with context
tracks = recommender.recommend(
    mood_tag="calm and focused",
    confidence=0.90,
    n_tracks=5,
    extra_context={
        "time_of_day": "evening",
        "activity": "reading",
        "preferences": "classical, ambient, instrumental",
        "recent_tracks": ["Weightless - Marconi Union", "Clair de Lune - Debussy"]
    },
    template=PromptTemplate.CONTEXTUAL
)
```

### Different Prompt Templates

```python
from src.llm_music_recommender import PromptTemplate

# 1. Basic - Simple, fast recommendations
tracks = recommender.recommend(
    mood_tag="happy",
    template=PromptTemplate.BASIC
)

# 2. Detailed - With reasoning
tracks = recommender.recommend(
    mood_tag="stressed and anxious",
    template=PromptTemplate.DETAILED
)

# 3. Contextual - Maximum context awareness
tracks = recommender.recommend(
    mood_tag="relaxed but focused",
    template=PromptTemplate.CONTEXTUAL,
    extra_context={"activity": "studying"}
)

# 4. Therapeutic - Evidence-based music therapy approach
tracks = recommender.recommend(
    mood_tag="sad and melancholic",
    template=PromptTemplate.THERAPEUTIC
)
```

## Complete Pipeline Example

Run the full end-to-end demo:

```bash
# With OpenAI API key
python examples/02_llm_recommendation_pipeline.py --mode simulated --api-key sk-...

# Using environment variable
export OPENAI_API_KEY="sk-..."
python examples/02_llm_recommendation_pipeline.py --mode simulated --n-trials 3

# With real EEG data
python examples/02_llm_recommendation_pipeline.py --mode deap --subject 1 --model gpt-4
```

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--mode` | Data source (`simulated`, `deap`, `real-time`) | `simulated` |
| `--api-key` | OpenAI API key | From `OPENAI_API_KEY` env var |
| `--model` | OpenAI model (`gpt-4o`, `gpt-4`, `gpt-3.5-turbo`) | `gpt-4o` |
| `--n-trials` | Number of EEG trials to process | `3` |
| `--tracks-per-trial` | Tracks to recommend per trial | `3` |
| `--subject` | DEAP subject ID | `1` |
| `--model-path` | Path to emotion recognition model | `None` |

## API Reference

### `LLMMusicRecommender`

**Constructor Parameters:**
- `api_key` (str, optional): OpenAI API key
- `model` (str): Model name ("gpt-4o", "gpt-4", "gpt-3.5-turbo")
- `temperature` (float): Creativity (0-2, default 0.7)
- `max_tokens` (int): Maximum response length (default 500)
- `enable_fallback` (bool): Use mock recommendations if API fails (default True)

**Methods:**

#### `recommend(mood_tag, confidence, n_tracks, extra_context, template)`
Generate music recommendations.

**Parameters:**
- `mood_tag` (str): Descriptive mood (e.g., "happy and energetic")
- `confidence` (float): Confidence score (0-1)
- `n_tracks` (int): Number of tracks to recommend
- `extra_context` (dict, optional): Additional context
- `template` (PromptTemplate): Prompt style to use

**Returns:** `List[LLMTrackRecommendation]`

#### `get_history()`
Get recommendation history.

**Returns:** `List[Dict]`

#### `clear_history()`
Clear recommendation history.

### `LLMTrackRecommendation`

**Attributes:**
- `title` (str): Song title
- `artist` (str): Artist name
- `reasoning` (str): Why this track fits
- `confidence` (float): Recommendation confidence (0-1)

**Methods:**
- `to_dict()`: Convert to dictionary
- `__str__()`: Returns "Artist - Title"

### Utility Functions

#### `emotion_to_mood_tag(emotion, confidence)`
Convert EmotionCategory to descriptive mood tag.

```python
tag = emotion_to_mood_tag(EmotionCategory.HAPPY, confidence=0.85)
# Returns: "happy and energetic"
```

#### `create_llm_recommender(api_key, model, **kwargs)`
Convenience function to create recommender.

```python
recommender = create_llm_recommender(api_key="sk-...", model="gpt-4o")
```

## Fallback Mode

When OpenAI API is unavailable (no key or network error), the system automatically falls back to curated recommendations:

```python
recommender = LLMMusicRecommender(enable_fallback=True)

# This works even without API key
tracks = recommender.recommend("happy", n_tracks=3)
# Returns: Curated tracks from fallback database
```

**Fallback Database Coverage:**
- Happy/Energetic moods
- Calm/Relaxed states
- Sad/Melancholic emotions
- Stressed/Anxious conditions
- Focused/Concentrated work

## Testing

### Run Unit Tests

```bash
# All tests
python tests/test_llm_recommender.py

# With live API (requires OPENAI_API_KEY)
export OPENAI_API_KEY="sk-..."
python tests/test_llm_recommender.py

# Using pytest
pytest tests/test_llm_recommender.py -v
```

### Test Coverage

✅ LLMTrackRecommendation dataclass  
✅ Recommender initialization  
✅ Fallback recommendations  
✅ Multiple mood handling  
✅ Prompt building  
✅ Response parsing (with/without reasoning)  
✅ History tracking  
✅ Context integration  
✅ Utility functions  
✅ Live API integration (if key available)  

## Performance

### Typical Latency (with GPT-4o)

| Component | Time |
|-----------|------|
| EEG Preprocessing | ~50ms |
| Feature Extraction | ~30ms |
| Emotion Detection | ~10ms |
| **LLM Recommendation** | **~800-1500ms** |
| **Total Pipeline** | **~900-1600ms** |

### Cost Estimate (GPT-4o)

- **Tokens per request:** ~200-400 tokens
- **Cost per request:** ~$0.001-0.003 USD
- **100 recommendations:** ~$0.10-0.30 USD

For high-volume applications, consider:
- Using GPT-3.5-turbo (cheaper, faster)
- Caching recommendations for similar moods
- Batch processing multiple recommendations

## Troubleshooting

### Issue: "No OpenAI API key found"

**Solution:**
```bash
export OPENAI_API_KEY="sk-your-key"
```

### Issue: "OpenAI library not available"

**Solution:**
```bash
pip install openai>=1.0.0
```

### Issue: Rate limit errors

**Solution:**
```python
recommender = LLMMusicRecommender(
    model="gpt-3.5-turbo",  # Cheaper/faster
    temperature=0.5          # Less variable
)
```

### Issue: Poor recommendations

**Solutions:**
1. Increase temperature for more creativity
2. Use CONTEXTUAL or DETAILED templates
3. Provide rich extra_context
4. Try different models (gpt-4 vs gpt-3.5-turbo)

## Best Practices

### 1. Use Context Richly

```python
# ❌ Basic
tracks = recommender.recommend("happy")

# ✅ Rich context
tracks = recommender.recommend(
    mood_tag="happy and energetic",
    confidence=0.85,
    extra_context={
        "time_of_day": "morning",
        "activity": "working out",
        "preferences": "electronic, pop, high tempo",
        "recent_tracks": ["Don't Stop Me Now - Queen"]
    }
)
```

### 2. Handle Errors Gracefully

```python
try:
    tracks = recommender.recommend(mood_tag, confidence)
except Exception as e:
    logger.error(f"LLM recommendation failed: {e}")
    # Fall back to traditional recommendation engine
    tracks = music_engine.recommend(emotion)
```

### 3. Cache Recommendations

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def get_cached_recommendations(mood_tag, n_tracks):
    return recommender.recommend(mood_tag, n_tracks=n_tracks)
```

### 4. Monitor API Usage

```python
# Track history
history = recommender.get_history()
print(f"API calls today: {len(history)}")
print(f"Estimated cost: ${len(history) * 0.002:.2f}")
```

## Future Enhancements

- [ ] User feedback loop (like/dislike tracks)
- [ ] Personalization based on listening history
- [ ] Multi-language prompt support
- [ ] Streaming responses for faster initial results
- [ ] Integration with Spotify API for track validation
- [ ] A/B testing different prompt templates
- [ ] Fine-tuned models for music recommendation

## License

Proprietary - Part of Neuro-Adaptive Music Player v2

## Author

Alexander V. - CMP9780M Assessment

## Support

For issues or questions:
1. Check the [troubleshooting section](#troubleshooting)
2. Run the unit tests: `python tests/test_llm_recommender.py`
3. Review examples: `examples/02_llm_recommendation_pipeline.py`
