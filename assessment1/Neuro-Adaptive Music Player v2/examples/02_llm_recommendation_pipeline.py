"""
LLM-Powered Music Recommendation Pipeline
==========================================

Complete end-to-end demonstration of EEG-to-music pipeline with
dynamic LLM-based recommendations:

EEG → Preprocessing → Feature Extraction → Emotion Detection →
LLM Mood Analysis → Dynamic Track Recommendations → Spotify Playback

This example shows how to integrate the LLM recommender into the
existing Neuro-Adaptive Music Player v2 architecture for creative,
context-aware music suggestions powered by GPT-4.

Usage:
    # With OpenAI API key
    python examples/02_llm_recommendation_pipeline.py --mode simulated --api-key sk-...
    
    # Using environment variable
    export OPENAI_API_KEY="sk-..."
    python examples/02_llm_recommendation_pipeline.py --mode simulated
    
    # With real EEG data
    python examples/02_llm_recommendation_pipeline.py --mode deap --subject 1

Author: Alexander V.
License: Proprietary
Version: 2.0.0
"""

import sys
import argparse
import logging
import time
from pathlib import Path
from typing import Tuple, List

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import v2 modules
from src.config import Config
from src.data_loaders import generate_simulated_data, load_deap, EEGDataset
from src.eeg_preprocessing import EEGPreprocessor
from src.eeg_features import EEGFeatureExtractor
from src.music_recommendation import EmotionCategory, MusicPlatform
from src.llm_music_recommender import (
    LLMMusicRecommender,
    LLMTrackRecommendation,
    emotion_to_mood_tag,
    PromptTemplate
)

# Try importing TensorFlow model
try:
    from src.emotion_recognition_model import EmotionRecognitionModel
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("NOTE: TensorFlow not available. Using mock emotion predictions.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# LLM-ENHANCED NEURO-ADAPTIVE MUSIC PLAYER
# =============================================================================

class LLMNeuroAdaptiveMusicPlayer:
    """
    Enhanced music player with LLM-powered dynamic recommendations.
    
    Extends the basic neuro-adaptive system with GPT-4 creativity for
    generating contextually-aware, emotionally-resonant music suggestions
    that go beyond static genre mappings.
    
    Pipeline:
        1. EEG Signal Acquisition
        2. Preprocessing (filtering, artifact removal)
        3. Feature Extraction (band powers, FAA, statistics)
        4. Emotion Recognition (CNN+BiLSTM or mock)
        5. **LLM Mood Analysis** (convert emotion → descriptive mood tag)
        6. **Dynamic Recommendation** (GPT-4 generates track suggestions)
        7. Spotify Playback (via existing music engine)
    """
    
    def __init__(
        self,
        config: Config,
        openai_api_key: str = None,
        llm_model: str = "gpt-4o",
        model_path: str = None,
        enable_spotify: bool = False
    ):
        """
        Initialize LLM-enhanced music player.
        
        Args:
            config: System configuration
            openai_api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            llm_model: LLM model to use (gpt-4o, gpt-4, gpt-3.5-turbo)
            model_path: Path to pre-trained emotion recognition model
            enable_spotify: Enable Spotify playback
        """
        logger.info("Initializing LLM-Enhanced Neuro-Adaptive Music Player...")
        
        self.config = config
        
        # Initialize preprocessing
        self.preprocessor = EEGPreprocessor(
            fs=config.SAMPLING_RATE,
            bandpass_low=config.BANDPASS_LOWCUT,
            bandpass_high=config.BANDPASS_HIGHCUT,
            notch_freq=config.NOTCH_FREQ
        )
        logger.info("✓ Preprocessor initialized")
        
        # Initialize feature extraction
        self.feature_extractor = EEGFeatureExtractor(
            fs=config.SAMPLING_RATE,
            bands=config.FREQ_BANDS
        )
        logger.info("✓ Feature extractor initialized")
        
        # Initialize emotion recognition model (optional)
        self.model = None
        if TENSORFLOW_AVAILABLE and model_path:
            try:
                self.model = EmotionRecognitionModel(
                    input_shape=(167,),
                    n_classes=5,
                    model_name="emotion_classifier"
                )
                self.model.load_model(model_path)
                logger.info(f"✓ Emotion model loaded from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load model: {e}. Using mock predictions.")
        else:
            logger.warning("⚠ TensorFlow not available or no model path. Using mock predictions.")
        
        # Initialize LLM recommender (KEY NEW COMPONENT)
        self.llm_recommender = LLMMusicRecommender(
            api_key=openai_api_key,
            model=llm_model,
            temperature=0.7,
            enable_fallback=True
        )
        logger.info(f"✓ LLM recommender initialized ({llm_model})")
        
        # Emotion label mapping
        self.emotion_labels = {
            0: EmotionCategory.CALM,
            1: EmotionCategory.NEUTRAL,
            2: EmotionCategory.HAPPY,
            3: EmotionCategory.SAD,
            4: EmotionCategory.STRESSED
        }
        
        logger.info("=" * 70)
        logger.info("✓ LLM-Enhanced Neuro-Adaptive Music Player Ready!")
        logger.info("=" * 70)
    
    def process_eeg_and_recommend(
        self,
        eeg_data: np.ndarray,
        n_tracks: int = 3,
        extra_context: dict = None,
        verbose: bool = True
    ) -> Tuple[EmotionCategory, float, List[LLMTrackRecommendation]]:
        """
        Complete pipeline: EEG → Emotion → LLM → Track Recommendations.
        
        Args:
            eeg_data: Raw EEG data (n_channels, n_samples)
            n_tracks: Number of tracks to recommend
            extra_context: Additional context for LLM (time, activity, etc.)
            verbose: Print detailed progress
            
        Returns:
            Tuple of (emotion, confidence, recommended_tracks)
        """
        if verbose:
            logger.info(f"\n{'='*70}")
            logger.info(f"Processing EEG Trial: {eeg_data.shape}")
            logger.info(f"{'='*70}")
        
        # =====================================================================
        # STEP 1: PREPROCESSING
        # =====================================================================
        start_time = time.time()
        preprocessed = self.preprocessor.preprocess(eeg_data)
        preprocess_time = (time.time() - start_time) * 1000
        
        if verbose:
            logger.info(f"[1/5] ✓ Preprocessing complete ({preprocess_time:.2f}ms)")
        
        # =====================================================================
        # STEP 2: FEATURE EXTRACTION
        # =====================================================================
        start_time = time.time()
        features_dict = self.feature_extractor.extract_all_features(preprocessed)
        features = self.feature_extractor.features_to_vector(features_dict, flatten=True)
        feature_time = (time.time() - start_time) * 1000
        
        if verbose:
            logger.info(f"[2/5] ✓ Feature extraction complete ({feature_time:.2f}ms)")
            logger.info(f"      → {features.shape[0]} features extracted")
        
        # =====================================================================
        # STEP 3: EMOTION RECOGNITION
        # =====================================================================
        start_time = time.time()
        features_batch = features.reshape(1, -1)
        
        if self.model is not None:
            # Use actual model
            prediction_probs = self.model.predict_proba(features_batch)[0]
            emotion_idx = int(np.argmax(prediction_probs))
            confidence = float(prediction_probs[emotion_idx])
        else:
            # Mock prediction
            emotion_idx = np.random.randint(0, len(self.emotion_labels))
            confidence = np.random.uniform(0.7, 0.95)
            prediction_probs = np.random.dirichlet(np.ones(len(self.emotion_labels)))
            if verbose:
                logger.warning("      ⚠ Using MOCK emotion prediction")
        
        emotion = self.emotion_labels.get(emotion_idx, EmotionCategory.NEUTRAL)
        predict_time = (time.time() - start_time) * 1000
        
        if verbose:
            logger.info(f"[3/5] ✓ Emotion recognition complete ({predict_time:.2f}ms)")
            logger.info(f"      → Detected: {emotion.value.upper()} (confidence: {confidence:.1%})")
        
        # =====================================================================
        # STEP 4: MOOD TAG GENERATION (for LLM)
        # =====================================================================
        start_time = time.time()
        mood_tag = emotion_to_mood_tag(emotion, confidence)
        mood_time = (time.time() - start_time) * 1000
        
        if verbose:
            logger.info(f"[4/5] ✓ Mood tag generated ({mood_time:.2f}ms)")
            logger.info(f"      → Mood: \"{mood_tag}\"")
        
        # =====================================================================
        # STEP 5: LLM DYNAMIC RECOMMENDATION (KEY STEP!)
        # =====================================================================
        start_time = time.time()
        
        # Enrich context with additional information
        context = extra_context or {}
        context["detected_emotion"] = emotion.value
        context["confidence_score"] = f"{confidence:.1%}"
        
        # Query LLM for creative recommendations
        recommended_tracks = self.llm_recommender.recommend(
            mood_tag=mood_tag,
            confidence=confidence,
            n_tracks=n_tracks,
            extra_context=context,
            template=PromptTemplate.CONTEXTUAL
        )
        
        llm_time = (time.time() - start_time) * 1000
        
        if verbose:
            logger.info(f"[5/5] ✓ LLM recommendations complete ({llm_time:.2f}ms)")
            logger.info(f"\n{'─'*70}")
            logger.info("🎵 RECOMMENDED TRACKS:")
            logger.info(f"{'─'*70}")
            for i, track in enumerate(recommended_tracks, 1):
                logger.info(f"  {i}. {track.artist} - {track.title}")
                if track.reasoning:
                    logger.info(f"     → {track.reasoning}")
            logger.info(f"{'─'*70}")
        
        total_time = preprocess_time + feature_time + predict_time + mood_time + llm_time
        if verbose:
            logger.info(f"\n⏱ Total Pipeline Time: {total_time:.2f}ms")
            logger.info(f"{'='*70}\n")
        
        return emotion, confidence, recommended_tracks
    
    def run_demo(
        self,
        dataset: EEGDataset,
        n_trials: int = 3,
        n_tracks_per_trial: int = 3
    ) -> None:
        """
        Run demonstration on dataset with LLM recommendations.
        
        Args:
            dataset: EEG dataset to process
            n_trials: Number of trials to process
            n_tracks_per_trial: Tracks to recommend per trial
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"LLM MUSIC RECOMMENDATION DEMO")
        logger.info(f"Dataset: {dataset.metadata.get('dataset', 'Unknown')}")
        logger.info(f"Trials: {n_trials}")
        logger.info(f"{'='*70}\n")
        
        all_recommendations = []
        
        for i in range(min(n_trials, len(dataset))):
            # Get trial data
            eeg_data, true_label = dataset[i]
            
            # Process and recommend
            emotion, confidence, tracks = self.process_eeg_and_recommend(
                eeg_data=eeg_data,
                n_tracks=n_tracks_per_trial,
                extra_context={
                    "trial_number": i + 1,
                    "activity": "relaxing"
                },
                verbose=True
            )
            
            all_recommendations.append({
                "trial": i + 1,
                "emotion": emotion.value,
                "confidence": confidence,
                "tracks": [str(t) for t in tracks]
            })
            
            # Brief pause between trials
            if i < n_trials - 1:
                time.sleep(1)
        
        # Summary
        logger.info(f"\n{'='*70}")
        logger.info("DEMO COMPLETE - SUMMARY")
        logger.info(f"{'='*70}")
        
        logger.info(f"\nProcessed {len(all_recommendations)} trials")
        logger.info(f"Total tracks recommended: {len(all_recommendations) * n_tracks_per_trial}")
        
        # Emotion distribution
        emotion_counts = {}
        for rec in all_recommendations:
            emotion = rec["emotion"]
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        logger.info(f"\nEmotion Distribution:")
        for emotion, count in emotion_counts.items():
            logger.info(f"  {emotion:12} : {count:2d} ({count/len(all_recommendations)*100:.0f}%)")
        
        avg_confidence = np.mean([r["confidence"] for r in all_recommendations])
        logger.info(f"\nAverage Confidence: {avg_confidence:.1%}")
        
        logger.info(f"\n{'='*70}")
        logger.info("✓ All recommendations generated by GPT-4 in real-time!")
        logger.info(f"{'='*70}\n")


# =============================================================================
# MAIN DEMONSTRATION
# =============================================================================

def main():
    """Main demonstration script."""
    parser = argparse.ArgumentParser(
        description="LLM-Powered Neuro-Adaptive Music Recommendations"
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['simulated', 'deap', 'real-time'],
        default='simulated',
        help='Data source mode'
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='OpenAI API key (or set OPENAI_API_KEY env var)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o',
        choices=['gpt-4o', 'gpt-4', 'gpt-3.5-turbo'],
        help='OpenAI model to use'
    )
    
    parser.add_argument(
        '--n-trials',
        type=int,
        default=3,
        help='Number of EEG trials to process'
    )
    
    parser.add_argument(
        '--tracks-per-trial',
        type=int,
        default=3,
        help='Number of tracks to recommend per trial'
    )
    
    parser.add_argument(
        '--subject',
        type=int,
        default=1,
        help='Subject ID for DEAP dataset'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to pre-trained emotion recognition model'
    )
    
    args = parser.parse_args()
    
    # Banner
    print("\n" + "="*70)
    print("       LLM-POWERED NEURO-ADAPTIVE MUSIC PLAYER V2")
    print("    Dynamic AI Recommendations via GPT-4 + EEG Emotions")
    print("="*70 + "\n")
    
    # Validate API key early
    import os
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("❌ OpenAI API key not found!")
        logger.info("💡 Set your API key in one of these ways:")
        logger.info("   1. Create .env file: cp .env.example .env")
        logger.info("   2. Set environment variable: export OPENAI_API_KEY='sk-...'")
        logger.info("   3. Pass via CLI: --api-key sk-...")
        logger.info("\n📖 See ENV_SETUP.md for detailed setup instructions")
        logger.info("📖 See SECURITY.md for API key security best practices")
        return 1
    
    # Validate API key format
    if not api_key.startswith('sk-'):
        logger.error(f"⚠️  API key format appears invalid (should start with 'sk-')")
        logger.info("💡 Verify your key at: https://platform.openai.com/api-keys")
        return 1
    
    # Initialize configuration
    config = Config()
    
    # Load dataset based on mode
    logger.info(f"Loading dataset: {args.mode}")
    
    if args.mode == 'simulated':
        dataset = generate_simulated_data(
            n_trials=args.n_trials,
            n_channels=32,
            duration=10.0,
            emotion='happy'
        )
        logger.info(f"✓ Generated {len(dataset)} simulated EEG trials")
    
    elif args.mode == 'deap':
        dataset = load_deap(
            subject=args.subject,
            data_dir=config.DATA_DIR / "DEAP"
        )
        logger.info(f"✓ Loaded DEAP subject {args.subject}: {len(dataset)} trials")
    
    else:
        logger.error(f"Mode '{args.mode}' not fully implemented yet")
        return
    
    # Initialize LLM-enhanced player with error handling
    try:
        player = LLMNeuroAdaptiveMusicPlayer(
            config=config,
            openai_api_key=api_key,
            llm_model=args.model,
            model_path=args.model_path,
            enable_spotify=False
        )
    except ValueError as e:
        logger.error(f"❌ Failed to initialize player: {e}")
        logger.info("💡 Check your OpenAI API key and try again")
        return 1
    except Exception as e:
        logger.error(f"❌ Unexpected error during initialization: {e}")
        return 1
    
    # Run demonstration with error handling
    try:
        player.run_demo(
            dataset=dataset,
            n_trials=args.n_trials,
            n_tracks_per_trial=args.tracks_per_trial
        )
    except KeyboardInterrupt:
        logger.info("\n⚠️  Demo interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"❌ Error during demo: {e}")
        logger.exception("Full traceback:")
        return 1
    
    # Tips
    print("\n" + "="*70)
    print("                    DEMO COMPLETE!")
    print("="*70)
    print("\nTips:")
    print("  • Set OPENAI_API_KEY environment variable for automatic auth")
    print("  • Use --model gpt-4 for more creative recommendations")
    print("  • Use --mode deap to test with real EEG data")
    print("  • All track suggestions are generated dynamically by AI!")
    print("\nNext steps:")
    print("  1. Integrate with Spotify API for actual playback")
    print("  2. Add user feedback loop to improve recommendations")
    print("  3. Experiment with different LLM prompt templates")
    print("  4. Train emotion model for better accuracy")
    print("\n")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
