"""
Unit Tests for LLM Music Recommender
=====================================

Tests the LLMMusicRecommender module with both live API calls
and fallback mode.

Run with:
    python -m pytest tests/test_llm_recommender.py -v
    
Or standalone:
    python tests/test_llm_recommender.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import unittest
from unittest.mock import Mock, patch
from src.llm_music_recommender import (
    LLMMusicRecommender,
    LLMTrackRecommendation,
    PromptTemplate,
    emotion_to_mood_tag,
    create_llm_recommender
)
from src.music_recommendation import EmotionCategory


class TestLLMTrackRecommendation(unittest.TestCase):
    """Test LLMTrackRecommendation dataclass."""
    
    def test_creation(self):
        """Test creating track recommendation."""
        track = LLMTrackRecommendation(
            title="Happy",
            artist="Pharrell Williams",
            reasoning="Upbeat and energetic",
            confidence=0.9
        )
        
        self.assertEqual(track.title, "Happy")
        self.assertEqual(track.artist, "Pharrell Williams")
        self.assertEqual(track.confidence, 0.9)
    
    def test_string_representation(self):
        """Test string conversion."""
        track = LLMTrackRecommendation(
            title="Happy",
            artist="Pharrell Williams"
        )
        
        self.assertEqual(str(track), "Pharrell Williams - Happy")
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        track = LLMTrackRecommendation(
            title="Happy",
            artist="Pharrell Williams",
            reasoning="Test",
            confidence=0.8
        )
        
        track_dict = track.to_dict()
        self.assertIn("title", track_dict)
        self.assertIn("artist", track_dict)
        self.assertIn("reasoning", track_dict)
        self.assertIn("confidence", track_dict)


class TestLLMMusicRecommender(unittest.TestCase):
    """Test LLMMusicRecommender class."""
    
    def test_initialization_with_fallback(self):
        """Test initializing recommender in fallback mode."""
        recommender = LLMMusicRecommender(
            api_key=None,
            enable_fallback=True
        )
        
        self.assertIsNotNone(recommender)
        self.assertTrue(recommender.enable_fallback)
    
    def test_fallback_recommendations(self):
        """Test generating fallback recommendations."""
        recommender = LLMMusicRecommender(
            api_key=None,
            enable_fallback=True
        )
        
        # Generate recommendations
        tracks = recommender.recommend(
            mood_tag="happy and energetic",
            confidence=0.85,
            n_tracks=3
        )
        
        # Verify results
        self.assertEqual(len(tracks), 3)
        self.assertIsInstance(tracks[0], LLMTrackRecommendation)
        self.assertTrue(all(t.title and t.artist for t in tracks))
    
    def test_multiple_moods(self):
        """Test recommendations for different moods."""
        recommender = LLMMusicRecommender(enable_fallback=True)
        
        moods = ["happy", "calm", "sad", "energetic", "relaxed"]
        
        for mood in moods:
            tracks = recommender.recommend(mood, n_tracks=2)
            self.assertEqual(len(tracks), 2)
            self.assertIsInstance(tracks[0], LLMTrackRecommendation)
    
    def test_prompt_building(self):
        """Test dynamic prompt construction."""
        recommender = LLMMusicRecommender(enable_fallback=True)
        
        # Build prompt
        prompt = recommender._build_prompt(
            mood_tag="happy and energetic",
            confidence=0.85,
            n_tracks=3,
            extra_context={"time_of_day": "morning", "activity": "working"},
            template=PromptTemplate.CONTEXTUAL
        )
        
        # Verify prompt contains key elements
        self.assertIn("happy and energetic", prompt)
        self.assertIn("morning", prompt)
        self.assertIn("working", prompt)
        self.assertIn("3", prompt)
    
    def test_response_parsing(self):
        """Test parsing LLM response text."""
        recommender = LLMMusicRecommender(enable_fallback=True)
        
        # Mock LLM response
        response_text = """1. Pharrell Williams - Happy
2. Katrina and the Waves - Walking on Sunshine
3. Mark Ronson ft. Bruno Mars - Uptown Funk"""
        
        # Parse
        tracks = recommender._parse_llm_response(response_text, expected_count=3)
        
        # Verify
        self.assertEqual(len(tracks), 3)
        self.assertEqual(tracks[0].artist, "Pharrell Williams")
        self.assertEqual(tracks[0].title, "Happy")
        self.assertEqual(tracks[2].artist, "Mark Ronson ft. Bruno Mars")
    
    def test_response_parsing_with_reasoning(self):
        """Test parsing response with reasoning."""
        recommender = LLMMusicRecommender(enable_fallback=True)
        
        response_text = """1. Enya - Only Time | Reason: Soothing and peaceful
2. Ludovico Einaudi - Nuvole Bianche | Reason: Gentle piano
3. Sigur Rós - Hoppípolla | Reason: Ethereal atmosphere"""
        
        tracks = recommender._parse_llm_response(response_text, expected_count=3)
        
        self.assertEqual(len(tracks), 3)
        self.assertIn("Soothing", tracks[0].reasoning)
        self.assertIn("piano", tracks[1].reasoning)
    
    def test_history_tracking(self):
        """Test recommendation history tracking."""
        recommender = LLMMusicRecommender(enable_fallback=True)
        
        # Make recommendations
        recommender.recommend("happy", n_tracks=2)
        recommender.recommend("calm", n_tracks=3)
        
        # Check history
        history = recommender.get_history()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["mood_tag"], "happy")
        self.assertEqual(history[1]["mood_tag"], "calm")
        
        # Clear history
        recommender.clear_history()
        self.assertEqual(len(recommender.get_history()), 0)
    
    def test_extra_context_integration(self):
        """Test using extra context in recommendations."""
        recommender = LLMMusicRecommender(enable_fallback=True)
        
        tracks = recommender.recommend(
            mood_tag="focused",
            confidence=0.9,
            n_tracks=2,
            extra_context={
                "time_of_day": "afternoon",
                "activity": "studying",
                "preferences": "classical, ambient"
            }
        )
        
        self.assertEqual(len(tracks), 2)
        
        # Verify context was recorded in history
        history = recommender.get_history()
        self.assertIn("preferences", history[0]["extra_context"])


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""
    
    def test_emotion_to_mood_tag(self):
        """Test emotion category to mood tag conversion."""
        # Test basic conversions
        tag = emotion_to_mood_tag(EmotionCategory.HAPPY, confidence=0.85)
        self.assertIn("happy", tag.lower())
        
        tag = emotion_to_mood_tag(EmotionCategory.CALM, confidence=0.9)
        self.assertIn("calm", tag.lower())
        
        # Test confidence modifiers
        tag_low = emotion_to_mood_tag(EmotionCategory.HAPPY, confidence=0.3)
        self.assertIn("somewhat", tag_low.lower())
        
        tag_high = emotion_to_mood_tag(EmotionCategory.EXCITED, confidence=0.95)
        self.assertIn("very", tag_high.lower())
    
    def test_create_llm_recommender(self):
        """Test convenience function."""
        recommender = create_llm_recommender(
            api_key=None,
            model="gpt-3.5-turbo"
        )
        
        self.assertIsInstance(recommender, LLMMusicRecommender)
        self.assertEqual(recommender.model, "gpt-3.5-turbo")


class TestLiveAPIIntegration(unittest.TestCase):
    """
    Integration tests with live OpenAI API.
    
    These tests are skipped if OPENAI_API_KEY is not set.
    """
    
    def setUp(self):
        """Check if API key is available."""
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            self.skipTest("OPENAI_API_KEY not set, skipping live API tests")
    
    def test_live_recommendation(self):
        """Test live API call (requires valid API key)."""
        recommender = LLMMusicRecommender(
            api_key=self.api_key,
            model="gpt-3.5-turbo",  # Use cheaper model for testing
            enable_fallback=True
        )
        
        tracks = recommender.recommend(
            mood_tag="happy and energetic",
            confidence=0.85,
            n_tracks=3
        )
        
        # Verify we got real recommendations
        self.assertEqual(len(tracks), 3)
        self.assertTrue(all(isinstance(t, LLMTrackRecommendation) for t in tracks))
        self.assertTrue(all(t.title and t.artist for t in tracks))
        
        print("\n✓ Live API Test - Recommended Tracks:")
        for i, track in enumerate(tracks, 1):
            print(f"  {i}. {track}")


def run_tests():
    """Run all tests."""
    # Discover and run tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # Run tests
    success = run_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
