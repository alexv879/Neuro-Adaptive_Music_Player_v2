"""
Complete Pipeline Example - Neuro-Adaptive Music Player v2

This script demonstrates the full end-to-end workflow:
1. Load/generate EEG data
2. Preprocess signals (filtering, artifact removal)
3. Extract features (band power, FAA, statistics)
4. Predict emotion using trained model
5. Recommend and play music based on emotion

This is a complete, self-contained demo using only v2 modules.

Usage:
    python examples/01_complete_pipeline.py --mode simulated
    python examples/01_complete_pipeline.py --mode deap --subject 1
    python examples/01_complete_pipeline.py --mode real-time

Author: Alexandru Emanuel Vasile
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
from src.data_loaders import generate_simulated_data, load_deap, load_seed, EEGDataset
from src.eeg_preprocessing import EEGPreprocessor
from src.eeg_features import EEGFeatureExtractor
from src.music_recommendation import (
    MusicRecommendationEngine,
    EmotionCategory,
    MusicPlatform
)

# Check if TensorFlow is available
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
    from src.emotion_recognition_model import EmotionRecognitionModel
except ImportError:
    TENSORFLOW_AVAILABLE = False
    EmotionRecognitionModel = None
    print("NOTE: TensorFlow not available. Using mock emotion predictions.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NeuroAdaptiveMusicPlayer:
    """
    Complete neuro-adaptive music player system.
    
    Integrates all v2 modules into a single pipeline for real-time
    emotion-based music recommendation.
    """
    
    def __init__(
        self,
        config: Config,
        model_path: str = None,
        music_platform: str = "none"
    ):
        """
        Initialize the complete system.
        
        Args:
            config: Configuration object
            model_path: Path to pre-trained emotion recognition model (optional)
            music_platform: Music platform ("spotify", "youtube", "local", "none")
        """
        logger.info("Initializing Neuro-Adaptive Music Player v2...")
        
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
        
        # Initialize emotion recognition model (if TensorFlow available)
        self.model = None
        if TENSORFLOW_AVAILABLE:
            self.model = EmotionRecognitionModel(
                input_shape=(167,),  # Adjust based on your feature count
                n_classes=5,
                model_name="emotion_classifier"
            )
            
            if model_path and Path(model_path).exists():
                self.model.load_model(model_path)
                logger.info(f"✓ Loaded pre-trained model from {model_path}")
            else:
                # Build a simple model for demo (would normally be pre-trained)
                self.model.build_model(architecture='dense')
                logger.info("✓ Built new model (untrained - for demo only)")
        else:
            logger.warning("⚠ TensorFlow not available. Using mock emotion predictions.")
        
        # Initialize music recommendation engine
        platform_enum = MusicPlatform(music_platform.lower())
        self.music_engine = MusicRecommendationEngine(platform=platform_enum)
        logger.info(f"✓ Music engine initialized ({music_platform})")
        
        # Emotion label mapping
        self.emotion_labels = {
            0: EmotionCategory.CALM,
            1: EmotionCategory.NEUTRAL,
            2: EmotionCategory.HAPPY,
            3: EmotionCategory.SAD,
            4: EmotionCategory.ANGRY
        }
        
        logger.info("=" * 60)
        logger.info("Neuro-Adaptive Music Player v2 - Ready!")
        logger.info("=" * 60)
    
    def process_trial(
        self,
        eeg_data: np.ndarray,
        channel_names: List[str] = None,
        verbose: bool = True
    ) -> Tuple[EmotionCategory, float, np.ndarray]:
        """
        Process a single EEG trial through the complete pipeline.

        Args:
            eeg_data: Raw EEG data (n_channels, n_samples)
            channel_names: List of channel names (required for FAA features)
            verbose: Print progress information

        Returns:
            Tuple of (emotion, confidence, features)
        """
        if verbose:
            logger.info(f"\nProcessing EEG trial: {eeg_data.shape}")

        # Use config channel names if not provided
        if channel_names is None:
            channel_names = self.config.CHANNEL_NAMES[:eeg_data.shape[0]]

        # Step 1: Preprocessing
        start_time = time.time()
        preprocessed = self.preprocessor.preprocess(eeg_data)
        preprocess_time = (time.time() - start_time) * 1000

        if verbose:
            logger.info(f"  [1/4] Preprocessing complete ({preprocess_time:.2f}ms)")

        # Step 2: Feature extraction (with channel names for FAA)
        start_time = time.time()
        features_dict = self.feature_extractor.extract_all_features(
            preprocessed,
            channel_names=channel_names,
            include_faa=True,
            include_stats=False,
            include_spectral=False
        )
        features = self.feature_extractor.features_to_vector(features_dict, flatten=True)
        feature_time = (time.time() - start_time) * 1000
        
        if verbose:
            logger.info(f"  [2/4] Feature extraction complete ({feature_time:.2f}ms)")
            logger.info(f"        Features shape: {features.shape}, Mean: {features.mean():.3f}")
        
        # Step 3: Emotion prediction
        start_time = time.time()
        features_batch = features.reshape(1, -1)  # Add batch dimension
        
        if self.model is not None:
            # Use actual model prediction
            prediction_probs = self.model.predict_proba(features_batch)[0]
            emotion_idx = int(np.argmax(prediction_probs))
            confidence = float(prediction_probs[emotion_idx])
        else:
            # Mock prediction when TensorFlow not available
            emotion_idx = np.random.randint(0, len(self.emotion_labels))
            confidence = np.random.uniform(0.6, 0.95)
            prediction_probs = np.random.dirichlet(np.ones(len(self.emotion_labels)))
            if verbose:
                logger.warning("        Using MOCK emotion prediction (no TensorFlow)")
        
        predict_time = (time.time() - start_time) * 1000
        
        emotion = self.emotion_labels.get(emotion_idx, EmotionCategory.NEUTRAL)
        
        if verbose:
            logger.info(f"  [3/4] Emotion prediction complete ({predict_time:.2f}ms)")
            logger.info(f"        Detected: {emotion.value} (confidence: {confidence:.2%})")
            logger.info(f"        All probs: {dict(zip(self.emotion_labels.values(), prediction_probs))}")
        
        # Step 4: Music recommendation
        start_time = time.time()
        track = self.music_engine.recommend(emotion, confidence=confidence)
        music_time = (time.time() - start_time) * 1000
        
        if verbose:
            logger.info(f"  [4/4] Music recommendation complete ({music_time:.2f}ms)")
            logger.info(f"        Recommended: {track}")
        
        total_time = preprocess_time + feature_time + predict_time + music_time
        if verbose:
            logger.info(f"\n  Total pipeline time: {total_time:.2f}ms")
        
        return emotion, confidence, features
    
    def run_demo(
        self,
        dataset: EEGDataset,
        n_trials: int = 5,
        auto_play: bool = False
    ) -> None:
        """
        Run demonstration on a dataset.
        
        Args:
            dataset: EEG dataset to process
            n_trials: Number of trials to process
            auto_play: If True, automatically play recommended music
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Running Demo: {n_trials} trials from {dataset.metadata.get('dataset', 'Unknown')}")
        logger.info(f"{'='*60}\n")
        
        results = []
        
        for i in range(min(n_trials, len(dataset))):
            logger.info(f"\n{'─'*60}")
            logger.info(f"Trial {i+1}/{n_trials}")
            logger.info(f"{'─'*60}")
            
            # Get trial data
            eeg_data, true_label = dataset[i]

            # Get channel names from dataset
            channel_names = dataset.channel_names if hasattr(dataset, 'channel_names') else None

            # Process through pipeline
            emotion, confidence, features = self.process_trial(
                eeg_data,
                channel_names=channel_names,
                verbose=True
            )
            
            # Recommend and optionally play music
            track = self.music_engine.recommend(emotion, confidence=confidence)
            
            if auto_play and self.music_engine.platform != MusicPlatform.NONE:
                logger.info(f"\n  ♫ Playing: {track}")
                success = self.music_engine.play(track)
                if success:
                    logger.info("  Playback started successfully")
                    time.sleep(3)  # Let it play for a bit
                    self.music_engine.pause()
                else:
                    logger.warning("  Playback failed")
            
            results.append({
                'trial': i,
                'emotion': emotion.value,
                'confidence': confidence,
                'track': str(track)
            })
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("Demo Complete - Summary")
        logger.info(f"{'='*60}")
        
        emotion_counts = {}
        for result in results:
            emotion = result['emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        logger.info(f"\nEmotion Distribution:")
        for emotion, count in emotion_counts.items():
            logger.info(f"  {emotion:12} : {count:2d} ({count/len(results)*100:.1f}%)")
        
        avg_confidence = np.mean([r['confidence'] for r in results])
        logger.info(f"\nAverage Confidence: {avg_confidence:.2%}")
        
        logger.info(f"\nMusic Engine Statistics:")
        stats = self.music_engine.get_statistics()
        logger.info(f"  Total tracks played: {stats['total_tracks']}")
        if 'emotions' in stats:
            logger.info(f"  Emotions: {stats['emotions']}")
        if 'platforms' in stats:
            logger.info(f"  Platforms: {stats['platforms']}")


# ==================== MAIN ====================

def main():
    """Main demonstration script."""
    parser = argparse.ArgumentParser(
        description="Neuro-Adaptive Music Player v2 - Complete Pipeline Demo"
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['simulated', 'deap', 'seed', 'real-time'],
        default='simulated',
        help='Data source mode'
    )
    
    parser.add_argument(
        '--subject',
        type=int,
        default=1,
        help='Subject ID for DEAP/SEED datasets'
    )
    
    parser.add_argument(
        '--n-trials',
        type=int,
        default=5,
        help='Number of trials to process'
    )
    
    parser.add_argument(
        '--music-platform',
        type=str,
        choices=['spotify', 'youtube', 'local', 'none'],
        default='none',
        help='Music streaming platform'
    )
    
    parser.add_argument(
        '--auto-play',
        action='store_true',
        help='Automatically play recommended music'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to pre-trained emotion recognition model'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "=" * 70)
    print(" " * 15 + "NEURO-ADAPTIVE MUSIC PLAYER V2")
    print(" " * 10 + "Complete Pipeline Demonstration")
    print("=" * 70 + "\n")
    
    # Initialize configuration
    config = Config()
    config.validate()
    
    # Initialize system
    player = NeuroAdaptiveMusicPlayer(
        config=config,
        model_path=args.model_path,
        music_platform=args.music_platform
    )
    
    # Load/generate data based on mode
    logger.info(f"\nLoading data (mode: {args.mode})...")
    
    if args.mode == 'simulated':
        # Generate simulated data
        logger.info("Generating simulated EEG data...")
        dataset = generate_simulated_data(
            n_trials=args.n_trials,
            n_channels=config.N_CHANNELS,
            sampling_rate=config.SAMPLING_RATE,
            duration=10.0,
            emotion=None  # Random emotions
        )
        logger.info(f"✓ Generated {len(dataset)} simulated trials")
    
    elif args.mode == 'deap':
        # Load DEAP dataset
        deap_path = "data/DEAP/"
        if not Path(deap_path).exists():
            logger.error(f"DEAP data not found at {deap_path}")
            logger.info("Please download DEAP dataset from:")
            logger.info("  https://www.eecs.qmul.ac.uk/mmv/datasets/deap/")
            sys.exit(1)
        
        logger.info(f"Loading DEAP subject {args.subject}...")
        dataset = load_deap(
            data_dir=deap_path,
            subject_ids=args.subject,
            eeg_only=True,
            label_type="valence_arousal"
        )
        logger.info(f"✓ Loaded DEAP data: {dataset}")
    
    elif args.mode == 'seed':
        # Load SEED dataset
        seed_path = "data/SEED/"
        if not Path(seed_path).exists():
            logger.error(f"SEED data not found at {seed_path}")
            logger.info("Please download SEED dataset from:")
            logger.info("  https://bcmi.sjtu.edu.cn/home/seed/")
            sys.exit(1)
        
        logger.info(f"Loading SEED subject {args.subject}, session 1...")
        dataset = load_seed(
            data_dir=seed_path,
            subject_id=args.subject,
            session=1
        )
        logger.info(f"✓ Loaded SEED data: {dataset}")
    
    elif args.mode == 'real-time':
        logger.error("Real-time mode not yet implemented")
        logger.info("Will be available when live_eeg_handler.py is complete")
        sys.exit(1)
    
    # Run demonstration
    player.run_demo(
        dataset=dataset,
        n_trials=args.n_trials,
        auto_play=args.auto_play
    )
    
    # Save music recommendation history
    if len(player.music_engine.history) > 0:
        history_file = f"music_cache/demo_history_{int(time.time())}.json"
        player.music_engine.save_history(history_file)
        logger.info(f"\n✓ Saved recommendation history to {history_file}")
    
    print("\n" + "=" * 70)
    print(" " * 20 + "DEMO COMPLETE!")
    print("=" * 70 + "\n")
    
    # Usage tips
    print("Tips:")
    print("  • Use --mode deap to load real EEG data")
    print("  • Use --music-platform spotify for real music playback")
    print("  • Use --auto-play to automatically play recommended music")
    print("  • Train a model first for accurate emotion predictions")
    print("\nNext steps:")
    print("  1. Train emotion recognition model (see examples/02_train_model.py)")
    print("  2. Set up Spotify API credentials for music playback")
    print("  3. Try with real EEG data from DEAP or SEED datasets")
    print()


if __name__ == "__main__":
    main()
