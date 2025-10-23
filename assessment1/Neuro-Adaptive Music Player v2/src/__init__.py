"""
Neuro-Adaptive Music Player v2

A production-quality EEG-based emotion recognition system for music recommendation.

Modules:
    - config: Configuration management
    - eeg_preprocessing: Signal preprocessing with artifact detection
    - eeg_features: Feature extraction (band power, FAA, statistics)
    - emotion_recognition_model: Deep learning models (CNN+BiLSTM)
    - data_loaders: Dataset loaders for DEAP/SEED (to be implemented)
    - live_eeg_handler: Real-time EEG streaming (to be implemented)
    - music_recommendation: Music selection based on emotions (to be implemented)
    - model_personalization: Transfer learning for user adaptation (to be implemented)
    - utils: Utility functions (to be implemented)

Example:
    >>> from src.eeg_preprocessing import EEGPreprocessor
    >>> from src.eeg_features import EEGFeatureExtractor
    >>> from src.emotion_recognition_model import EmotionRecognitionModel
    >>>
    >>> # Preprocessing
    >>> preprocessor = EEGPreprocessor(sampling_rate=256)
    >>> cleaned_data = preprocessor.preprocess(raw_eeg_data)
    >>>
    >>> # Feature extraction
    >>> extractor = EEGFeatureExtractor(sampling_rate=256)
    >>> features = extractor.extract_features(cleaned_data)
    >>>
    >>> # Emotion prediction
    >>> model = EmotionRecognitionModel(input_shape=(167,))
    >>> model.load_model('models/pretrained/deap_cnn_bilstm.h5')
    >>> emotion = model.predict(features)

Author: Alexander V.
License: See LICENSE file
Version: 2.0.0
"""

__version__ = '2.0.0'
__author__ = 'Alexander V.'
__license__ = 'Proprietary'

# Import main classes for convenience
try:
    from .config import Config
    from .eeg_preprocessing import EEGPreprocessor
    from .eeg_features import EEGFeatureExtractor
    from .emotion_recognition_model import EmotionRecognitionModel
    from .data_loaders import (
        DEAPLoader,
        SEEDLoader,
        SimulatedEEGGenerator,
        EEGDataset,
        load_deap,
        load_seed,
        generate_simulated_data
    )
    from .music_recommendation import (
        MusicRecommendationEngine,
        EmotionCategory,
        MusicPlatform,
        Track,
        create_recommendation_engine
    )
    from .llm_music_recommender import (
        LLMMusicRecommender,
        LLMTrackRecommendation,
        PromptTemplate,
        create_llm_recommender,
        emotion_to_mood_tag
    )
    
    __all__ = [
        # Core processing
        'Config',
        'EEGPreprocessor',
        'EEGFeatureExtractor',
        'EmotionRecognitionModel',
        # Data loading
        'DEAPLoader',
        'SEEDLoader',
        'SimulatedEEGGenerator',
        'EEGDataset',
        'load_deap',
        'load_seed',
        'generate_simulated_data',
        # Music recommendation
        'MusicRecommendationEngine',
        'EmotionCategory',
        'MusicPlatform',
        'Track',
        'create_recommendation_engine',
        # LLM-powered recommendations
        'LLMMusicRecommender',
        'LLMTrackRecommendation',
        'PromptTemplate',
        'create_llm_recommender',
        'emotion_to_mood_tag',
    ]
except ImportError as e:
    # Modules not yet available
    __all__ = []
    import warnings
    warnings.warn(f"Failed to import some modules: {e}")

# Package-level configuration
import warnings
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Suppress TensorFlow warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

try:
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO/WARNING
except:
    pass
