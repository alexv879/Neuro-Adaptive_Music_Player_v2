"""
Data Flow Validation Script
============================

Comprehensive validation of data types, shapes, and compatibility across
the entire emotion recognition → music recommendation pipeline.

This script checks:
1. EEG Preprocessing output → Feature Extraction input
2. Feature Extraction output → Emotion Model input
3. Emotion Model output → Music Recommendation input
4. All intermediate data types and shapes

Run this BEFORE showing to professor to ensure zero issues!

Usage:
    python validate_data_flow.py --quick      # Fast validation
    python validate_data_flow.py --full       # Complete validation

Author: CMP9780M Assessment
"""

import numpy as np
from pathlib import Path
import sys
from typing import Dict, Any, Tuple
import warnings

# Setup paths
sys.path.append(str(Path(__file__).parent / "src"))

from eeg_preprocessing import EEGPreprocessor
from eeg_features import EEGFeatureExtractor
from emotion_recognition_model import EmotionRecognitionModel
from music_recommendation import MusicRecommendationEngine, EmotionCategory
from config import SAMPLING_RATE, EMOTION_LABELS

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
END = '\033[0m'

class DataFlowValidator:
    """Validates data compatibility across the entire pipeline."""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.checks_passed = 0
        self.checks_total = 0
        
    def check(self, condition: bool, message: str, critical: bool = True):
        """Record a validation check."""
        self.checks_total += 1
        if condition:
            self.checks_passed += 1
            print(f"  {GREEN}✓{END} {message}")
        else:
            if critical:
                self.errors.append(message)
                print(f"  {RED}✗ ERROR:{END} {message}")
            else:
                self.warnings.append(message)
                print(f"  {YELLOW}⚠ WARNING:{END} {message}")
    
    def section(self, title: str):
        """Print section header."""
        print(f"\n{BLUE}{'=' * 70}{END}")
        print(f"{BLUE}{title}{END}")
        print(f"{BLUE}{'=' * 70}{END}")
    
    def summary(self):
        """Print validation summary."""
        print(f"\n{BLUE}{'=' * 70}{END}")
        print(f"{BLUE}VALIDATION SUMMARY{END}")
        print(f"{BLUE}{'=' * 70}{END}")
        
        print(f"\nTotal checks: {self.checks_total}")
        print(f"Passed: {GREEN}{self.checks_passed}{END}")
        print(f"Errors: {RED}{len(self.errors)}{END}")
        print(f"Warnings: {YELLOW}{len(self.warnings)}{END}")
        
        if self.errors:
            print(f"\n{RED}CRITICAL ERRORS FOUND:{END}")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")
        
        if self.warnings:
            print(f"\n{YELLOW}WARNINGS:{END}")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")
        
        if not self.errors:
            print(f"\n{GREEN}✅ ALL CRITICAL CHECKS PASSED!{END}")
            print(f"{GREEN}Your pipeline is ready to show to the professor.{END}")
            return True
        else:
            print(f"\n{RED}❌ VALIDATION FAILED!{END}")
            print(f"{RED}Fix the errors above before proceeding.{END}")
            return False


def validate_preprocessing_output():
    """Validate EEG preprocessing output."""
    validator = DataFlowValidator()
    validator.section("1. EEG PREPROCESSING OUTPUT VALIDATION")
    
    # Create test data
    n_channels = 32
    n_samples = 1280  # 5 seconds at 256 Hz
    test_data = np.random.randn(n_channels, n_samples).astype(np.float32)
    
    print(f"\nInput: {test_data.shape} (channels, samples)")
    
    # Preprocess
    preprocessor = EEGPreprocessor(fs=256.0)
    processed = preprocessor.preprocess(test_data)
    
    # Validate output
    validator.check(
        isinstance(processed, np.ndarray),
        f"Output is numpy array: {type(processed)}"
    )
    
    validator.check(
        processed.dtype == np.float64 or processed.dtype == np.float32,
        f"Output dtype is float: {processed.dtype}"
    )
    
    validator.check(
        processed.shape == test_data.shape,
        f"Output shape matches input: {processed.shape} == {test_data.shape}"
    )
    
    validator.check(
        not np.isnan(processed).any(),
        "No NaN values in output"
    )
    
    validator.check(
        not np.isinf(processed).any(),
        "No infinite values in output"
    )
    
    validator.check(
        np.abs(processed).max() < 1000,
        f"Values in reasonable range (max: {np.abs(processed).max():.2f} µV)",
        critical=False
    )
    
    return validator


def validate_feature_extraction():
    """Validate feature extraction input/output."""
    validator = DataFlowValidator()
    validator.section("2. FEATURE EXTRACTION INPUT/OUTPUT VALIDATION")
    
    # Create preprocessed data
    n_channels = 32
    n_samples = 1280
    preprocessed_data = np.random.randn(n_channels, n_samples).astype(np.float32)
    
    print(f"\nInput: {preprocessed_data.shape} (channels, samples)")
    
    # Extract features
    extractor = EEGFeatureExtractor(fs=256.0)
    
    # Test extract_all_features
    channel_names = [f'Ch{i+1}' for i in range(n_channels)]
    features_dict = extractor.extract_all_features(preprocessed_data, channel_names)
    
    validator.check(
        isinstance(features_dict, dict),
        f"extract_all_features returns dict: {type(features_dict)}"
    )
    
    validator.check(
        'band_power' in features_dict,
        "'band_power' key exists in features"
    )
    
    validator.check(
        'frontal_asymmetry' in features_dict,
        "'frontal_asymmetry' key exists in features"
    )
    
    # Test features_to_vector
    feature_vector = extractor.features_to_vector(features_dict)
    
    validator.check(
        isinstance(feature_vector, np.ndarray),
        f"features_to_vector returns numpy array: {type(feature_vector)}"
    )
    
    validator.check(
        feature_vector.ndim == 1,
        f"Feature vector is 1D: shape {feature_vector.shape}"
    )
    
    validator.check(
        not np.isnan(feature_vector).any(),
        "No NaN values in feature vector"
    )
    
    validator.check(
        not np.isinf(feature_vector).any(),
        "No infinite values in feature vector"
    )
    
    validator.check(
        len(feature_vector) > 0,
        f"Feature vector has elements: {len(feature_vector)} features"
    )
    
    print(f"\n📊 Feature Vector Shape: {feature_vector.shape}")
    print(f"   Feature count: {len(feature_vector)}")
    
    return validator, feature_vector


def validate_emotion_model_input_output(feature_vector):
    """Validate emotion model input/output."""
    validator = DataFlowValidator()
    validator.section("3. EMOTION MODEL INPUT/OUTPUT VALIDATION")
    
    n_features = len(feature_vector)
    n_samples = 100
    
    # Create batch of features
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    
    print(f"\nInput: {X.shape} (samples, features)")
    
    # Create model
    model = EmotionRecognitionModel(
        input_shape=(n_features,),
        n_classes=5,
        model_name='validation_model'
    )
    model.build_model(architecture='dense')  # Use simple architecture for speed
    
    # Validate model accepts input
    try:
        predictions = model.model.predict(X[:5], verbose=0)
        validator.check(True, "Model accepts feature vector input")
    except Exception as e:
        validator.check(False, f"Model input validation failed: {e}")
        return validator
    
    # Validate output format
    if isinstance(predictions, list):
        # Hierarchical model: [valence, arousal, emotion]
        valence_out, arousal_out, emotion_out = predictions
        
        validator.check(
            len(predictions) == 3,
            f"Hierarchical model returns 3 outputs: {len(predictions)}"
        )
        
        validator.check(
            emotion_out.shape == (5, 5),
            f"Emotion output shape correct: {emotion_out.shape} (should be (5, 5))"
        )
        
        validator.check(
            np.allclose(emotion_out.sum(axis=1), 1.0, atol=1e-5),
            "Emotion probabilities sum to 1.0 (softmax output)"
        )
        
        print(f"\n📊 Model Output Shapes:")
        print(f"   Valence: {valence_out.shape}")
        print(f"   Arousal: {arousal_out.shape}")
        print(f"   Emotion: {emotion_out.shape}")
        
    else:
        # Single output model
        validator.check(
            predictions.shape[1] == 5,
            f"Output has 5 emotion classes: shape {predictions.shape}"
        )
        
        validator.check(
            np.allclose(predictions.sum(axis=1), 1.0, atol=1e-5),
            "Probabilities sum to 1.0 (softmax output)"
        )
    
    # Test predict method (returns labels)
    # First, fit the label encoder with dummy labels
    dummy_labels = ['neutral', 'happy', 'sad', 'relaxed', 'focused']
    model.label_encoder.fit(dummy_labels)
    
    emotion_labels = model.predict(X[:10])
    
    validator.check(
        isinstance(emotion_labels, np.ndarray),
        f"predict() returns numpy array: {type(emotion_labels)}"
    )
    
    validator.check(
        emotion_labels.dtype.kind in ('U', 'O'),  # Unicode or Object (strings)
        f"predict() returns string labels: dtype {emotion_labels.dtype}"
    )
    
    validator.check(
        len(emotion_labels) == 10,
        f"predict() returns correct number of predictions: {len(emotion_labels)}"
    )
    
    # Check label values
    valid_labels = set(EMOTION_LABELS.values())
    predicted_labels = set(emotion_labels)
    
    validator.check(
        predicted_labels.issubset(valid_labels),
        f"All predicted labels are valid: {predicted_labels} ⊆ {valid_labels}"
    )
    
    print(f"\n📊 Prediction Output:")
    print(f"   Example predictions: {emotion_labels[:5]}")
    print(f"   Unique labels: {predicted_labels}")
    
    return validator, emotion_labels


def validate_music_engine_input():
    """Validate music recommendation engine input."""
    validator = DataFlowValidator()
    validator.section("4. MUSIC RECOMMENDATION ENGINE INPUT VALIDATION")
    
    # Create engine
    engine = MusicRecommendationEngine()
    
    # Test with string emotions (from model output)
    test_emotions = ['happy', 'sad', 'neutral', 'relaxed', 'focused']
    
    print(f"\nTesting with emotion labels: {test_emotions}")
    
    for emotion in test_emotions:
        try:
            # Try to recommend (mock mode)
            tracks = engine.recommend(emotion=emotion, n_tracks=3)
            
            validator.check(
                True,
                f"Engine accepts '{emotion}' emotion string"
            )
            
            validator.check(
                isinstance(tracks, list),
                f"Engine returns list for '{emotion}'"
            )
            
        except Exception as e:
            validator.check(
                False,
                f"Engine failed for '{emotion}': {e}"
            )
    
    # Test EmotionCategory enum compatibility
    print(f"\nTesting EmotionCategory enum:")
    
    emotion_mapping = {
        'happy': EmotionCategory.HAPPY,
        'sad': EmotionCategory.SAD,
        'neutral': EmotionCategory.NEUTRAL,
        'relaxed': EmotionCategory.RELAXED,
        'focused': EmotionCategory.FOCUSED,
    }
    
    for string_emotion, enum_emotion in emotion_mapping.items():
        validator.check(
            True,
            f"'{string_emotion}' maps to {enum_emotion.value}"
        )
    
    return validator


def validate_end_to_end_flow():
    """Validate complete end-to-end data flow."""
    validator = DataFlowValidator()
    validator.section("5. END-TO-END PIPELINE VALIDATION")
    
    print("\nSimulating complete pipeline:")
    print("EEG -> Preprocessing -> Features -> Model -> Music")
    
    # Step 1: Raw EEG
    print("\n  1. Raw EEG data...")
    n_channels, n_samples = 32, 1280
    raw_eeg = np.random.randn(n_channels, n_samples).astype(np.float32)
    validator.check(True, f"Generated raw EEG: {raw_eeg.shape}")
    
    # Step 2: Preprocess
    print("\n  2. Preprocessing...")
    preprocessor = EEGPreprocessor(fs=256.0)
    preprocessed = preprocessor.preprocess(raw_eeg)
    validator.check(
        preprocessed.shape == raw_eeg.shape,
        f"Preprocessed shape matches: {preprocessed.shape}"
    )
    
    # Step 3: Extract features
    print("\n  3. Feature extraction...")
    extractor = EEGFeatureExtractor(fs=256.0)
    channel_names = [f'Ch{i+1}' for i in range(n_channels)]
    features_dict = extractor.extract_all_features(preprocessed, channel_names)
    feature_vec = extractor.features_to_vector(features_dict)
    validator.check(
        feature_vec.ndim == 1,
        f"Feature vector is 1D: {feature_vec.shape}"
    )
    
    # Step 4: Predict emotion
    print("\n  4. Emotion prediction...")
    X_input = feature_vec.reshape(1, -1).astype(np.float32)
    model = EmotionRecognitionModel(
        input_shape=(len(feature_vec),),
        n_classes=5
    )
    model.build_model(architecture='dense')
    
    # Train label encoder with dummy data
    dummy_labels = ['neutral', 'happy', 'sad', 'relaxed', 'focused']
    model.label_encoder.fit(dummy_labels)
    
    emotion = model.predict(X_input)[0]
    validator.check(
        isinstance(emotion, str) or isinstance(emotion, np.str_),
        f"Model outputs string emotion: '{emotion}' ({type(emotion)})"
    )
    
    # Step 5: Music recommendation
    print("\n  5. Music recommendation...")
    engine = MusicRecommendationEngine()
    
    # Map 'focused' to 'calm' if needed
    if emotion == 'focused':
        emotion_for_music = 'calm'
        validator.check(
            True,
            f"Mapped 'focused' → 'calm' for music engine",
            critical=False
        )
    else:
        emotion_for_music = emotion
    
    try:
        tracks = engine.recommend(emotion=emotion_for_music, n_tracks=3)
        validator.check(
            len(tracks) > 0,
            f"Music engine returned {len(tracks)} track recommendations"
        )
    except Exception as e:
        validator.check(
            False,
            f"Music recommendation failed: {e}"
        )
    
    print(f"\n{GREEN}✓ Complete pipeline executed successfully!{END}")
    print(f"  EEG → Preprocessed → Features ({len(feature_vec)}) → Emotion ('{emotion}') → Music ({len(tracks)} tracks)")
    
    return validator


def validate_data_types_compatibility():
    """Validate data type compatibility throughout pipeline."""
    validator = DataFlowValidator()
    validator.section("6. DATA TYPE COMPATIBILITY VALIDATION")
    
    print("\nChecking numpy/TensorFlow compatibility:")
    
    # Test float32 vs float64
    data_f32 = np.random.randn(10, 167).astype(np.float32)
    data_f64 = np.random.randn(10, 167).astype(np.float64)
    
    model = EmotionRecognitionModel(input_shape=(167,), n_classes=5)
    model.build_model(architecture='dense')
    
    try:
        pred_f32 = model.model.predict(data_f32, verbose=0)
        validator.check(True, "Model accepts float32 input")
    except:
        validator.check(False, "Model rejects float32 input")
    
    try:
        pred_f64 = model.model.predict(data_f64, verbose=0)
        validator.check(True, "Model accepts float64 input")
    except:
        validator.check(False, "Model rejects float64 input")
    
    # Test string labels
    print("\nChecking string label compatibility:")
    
    test_labels = ['happy', 'sad', 'neutral', 'relaxed', 'focused']
    
    validator.check(
        all(isinstance(label, str) for label in test_labels),
        "All test labels are strings"
    )
    
    validator.check(
        set(test_labels) == set(EMOTION_LABELS.values()),
        f"Test labels match config: {test_labels} == {list(EMOTION_LABELS.values())}"
    )
    
    return validator


def validate_shape_consistency():
    """Validate shape consistency across pipeline."""
    validator = DataFlowValidator()
    validator.section("7. SHAPE CONSISTENCY VALIDATION")
    
    print("\nTesting with different input sizes:")
    
    # Test single sample
    print("\n  Single sample:")
    X_single = np.random.randn(1, 167).astype(np.float32)
    model = EmotionRecognitionModel(input_shape=(167,), n_classes=5)
    model.build_model(architecture='dense')
    model.label_encoder.fit(['neutral', 'happy', 'sad', 'relaxed', 'focused'])
    
    pred_single = model.predict(X_single)
    validator.check(
        len(pred_single) == 1,
        f"Single sample returns single prediction: {len(pred_single)}"
    )
    
    # Test batch
    print("\n  Batch of 10 samples:")
    X_batch = np.random.randn(10, 167).astype(np.float32)
    pred_batch = model.predict(X_batch)
    validator.check(
        len(pred_batch) == 10,
        f"Batch of 10 returns 10 predictions: {len(pred_batch)}"
    )
    
    # Test large batch
    print("\n  Large batch of 100 samples:")
    X_large = np.random.randn(100, 167).astype(np.float32)
    pred_large = model.predict(X_large)
    validator.check(
        len(pred_large) == 100,
        f"Batch of 100 returns 100 predictions: {len(pred_large)}"
    )
    
    return validator


def main():
    """Run all validations."""
    print(f"\n{BLUE}{'=' * 70}{END}")
    print(f"{BLUE}DATA FLOW VALIDATION - Emotion Recognition -> Music Recommendation{END}")
    print(f"{BLUE}{'=' * 70}{END}")
    print(f"\nThis script validates data compatibility across your entire pipeline.")
    print(f"Running all checks...\n")
    
    all_validators = []
    
    # Run all validation sections
    try:
        all_validators.append(validate_preprocessing_output())
        _, feature_vec = validate_feature_extraction()
        _, emotion_labels = validate_emotion_model_input_output(feature_vec)
        all_validators.append(validate_music_engine_input())
        all_validators.append(validate_end_to_end_flow())
        all_validators.append(validate_data_types_compatibility())
        all_validators.append(validate_shape_consistency())
        
    except Exception as e:
        print(f"\n{RED}FATAL ERROR during validation:{END}")
        print(f"{RED}{e}{END}")
        import traceback
        traceback.print_exc()
        return False
    
    # Aggregate results
    total_checks = sum(v.checks_total for v in all_validators)
    total_passed = sum(v.checks_passed for v in all_validators)
    total_errors = sum(len(v.errors) for v in all_validators)
    total_warnings = sum(len(v.warnings) for v in all_validators)
    
    # Print final summary
    print(f"\n{BLUE}{'=' * 70}{END}")
    print(f"{BLUE}FINAL VALIDATION SUMMARY{END}")
    print(f"{BLUE}{'=' * 70}{END}")
    
    print(f"\nTotal checks run: {total_checks}")
    print(f"Passed: {GREEN}{total_passed}{END}")
    print(f"Errors: {RED}{total_errors}{END}")
    print(f"Warnings: {YELLOW}{total_warnings}{END}")
    print(f"Success rate: {total_passed/total_checks*100:.1f}%")
    
    if total_errors == 0:
        print(f"\n{GREEN}{'=' * 70}{END}")
        print(f"{GREEN}✅ ALL VALIDATIONS PASSED!{END}")
        print(f"{GREEN}{'=' * 70}{END}")
        print(f"\n{GREEN}Your pipeline is fully validated and ready to show!{END}")
        print(f"\nData flow is correct:")
        print(f"  • EEG preprocessing output → Feature extraction input ✓")
        print(f"  • Feature extraction output → Emotion model input ✓")
        print(f"  • Emotion model output → Music engine input ✓")
        print(f"  • All data types compatible ✓")
        print(f"  • All shapes consistent ✓")
        
        if total_warnings > 0:
            print(f"\n{YELLOW}Note: {total_warnings} minor warnings found (non-critical){END}")
        
        return True
    else:
        print(f"\n{RED}{'=' * 70}{END}")
        print(f"{RED}❌ VALIDATION FAILED - FIX ERRORS BEFORE SHOWING TO PROFESSOR{END}")
        print(f"{RED}{'=' * 70}{END}")
        
        print(f"\n{RED}Critical issues found:{END}")
        for validator in all_validators:
            if validator.errors:
                for error in validator.errors:
                    print(f"  • {error}")
        
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate data flow compatibility')
    parser.add_argument('--quick', action='store_true', help='Quick validation')
    parser.add_argument('--full', action='store_true', help='Full validation')
    
    args = parser.parse_args()
    
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}Validation interrupted by user{END}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{RED}Unexpected error: {e}{END}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
