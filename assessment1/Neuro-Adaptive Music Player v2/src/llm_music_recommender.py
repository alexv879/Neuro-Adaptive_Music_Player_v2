"""
LLM Music Recommender - Dynamic AI-Powered Music Recommendations
=================================================================

Uses OpenAI GPT models to generate context-aware music recommendations
based on real-time EEG-detected emotions and moods. Provides dynamic,
creative suggestions that adapt to user state, time of day, and preferences.

Key Features:
- Dynamic prompt construction from emotion tags and confidence scores
- OpenAI GPT-4/GPT-4o integration with fallback to GPT-3.5-turbo
- Context-aware recommendations (time of day, user history, preferences)
- Structured output parsing for reliable track extraction
- Graceful fallback when API unavailable
- Integration with Spotify/YouTube/local playback via existing music engine

Author: Alexander V.
License: Proprietary
Version: 2.0.0
"""

from __future__ import annotations
import os
import logging
import re
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

# OpenAI integration with graceful fallback
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("WARNING: OpenAI library not available. Install with: pip install openai>=1.0.0")

# Import emotion categories from existing music recommendation module
try:
    from .music_recommendation import EmotionCategory, MusicPlatform
except ImportError:
    # Fallback definitions for standalone use
    from enum import Enum
    class EmotionCategory(Enum):
        CALM = "calm"
        HAPPY = "happy"
        SAD = "sad"
        ANGRY = "angry"
        NEUTRAL = "neutral"
        EXCITED = "excited"
        RELAXED = "relaxed"
        STRESSED = "stressed"

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class LLMTrackRecommendation:
    """
    Represents a track recommended by the LLM.
    
    Attributes:
        title: Song title
        artist: Artist name
        reasoning: LLM's explanation for why this track fits the mood
        confidence: Recommendation confidence (0-1)
    """
    title: str
    artist: str
    reasoning: str = ""
    confidence: float = 1.0
    
    def __str__(self) -> str:
        return f"{self.artist} - {self.title}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "title": self.title,
            "artist": self.artist,
            "reasoning": self.reasoning,
            "confidence": self.confidence
        }


class PromptTemplate(Enum):
    """Pre-defined prompt templates for different recommendation scenarios."""
    
    BASIC = "basic"
    DETAILED = "detailed"
    CONTEXTUAL = "contextual"
    THERAPEUTIC = "therapeutic"


# =============================================================================
# LLM MUSIC RECOMMENDER
# =============================================================================

class LLMMusicRecommender:
    """
    Dynamic music recommendation engine powered by OpenAI LLMs.
    
    Uses GPT-4 or GPT-3.5-turbo to generate creative, context-aware music
    recommendations based on real-time EEG emotion detection.
    
    Features:
    - Dynamic prompt construction from emotion tags and confidence
    - Context enrichment (time of day, user preferences, history)
    - Structured output parsing with robust error handling
    - API rate limiting and error recovery
    - Fallback to mock recommendations when API unavailable
    
    Example:
        >>> recommender = LLMMusicRecommender(api_key="sk-...")
        >>> tracks = recommender.recommend(
        ...     mood_tag="happy and energetic",
        ...     confidence=0.85,
        ...     extra_context={"time_of_day": "morning"}
        ... )
        >>> for track in tracks:
        ...     print(f"{track.artist} - {track.title}")
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 500,
        enable_fallback: bool = True
    ):
        """
        Initialize LLM music recommender.
        
        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: Model to use (gpt-4o, gpt-4, gpt-3.5-turbo)
            temperature: Creativity level (0-2, higher = more creative)
            max_tokens: Maximum tokens in response
            enable_fallback: Use mock recommendations if API unavailable
            
        Raises:
            ValueError: If OpenAI not available and fallback disabled
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_fallback = enable_fallback
        
        # Initialize OpenAI client
        if OPENAI_AVAILABLE:
            # Get API key from parameter, env var, or config
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
            
            if self.api_key:
                try:
                    self.client = OpenAI(api_key=self.api_key)
                    self._test_connection()
                    logger.info(f"✓ LLM recommender initialized with {model}")
                except Exception as e:
                    logger.warning(f"OpenAI initialization failed: {e}")
                    self.client = None
                    if not enable_fallback:
                        raise
            else:
                logger.warning("No OpenAI API key found. Set OPENAI_API_KEY environment variable.")
                self.client = None
                if not enable_fallback:
                    raise ValueError("OpenAI API key required when fallback disabled")
        else:
            self.client = None
            if not enable_fallback:
                raise ValueError("OpenAI library not installed and fallback disabled")
            logger.warning("OpenAI library not available. Using fallback recommendations.")
        
        # Track recommendation history
        self.recommendation_history: List[Dict[str, Any]] = []
    
    def _test_connection(self) -> bool:
        """Test OpenAI API connection."""
        try:
            # Simple test call
            response = self.client.chat.completions.create(
                model=self.model if self.model != "gpt-4o" else "gpt-3.5-turbo",  # Fallback for testing
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            return True
        except Exception as e:
            logger.warning(f"OpenAI connection test failed: {e}")
            return False
    
    def recommend(
        self,
        mood_tag: str,
        confidence: float = 1.0,
        n_tracks: int = 3,
        extra_context: Optional[Dict[str, Any]] = None,
        template: PromptTemplate = PromptTemplate.CONTEXTUAL
    ) -> List[LLMTrackRecommendation]:
        """
        Generate music recommendations using LLM based on detected mood.
        
        Args:
            mood_tag: Emotion/mood description (e.g., "happy and energetic", "calm")
            confidence: Confidence score from emotion detection (0-1)
            n_tracks: Number of tracks to recommend
            extra_context: Additional context (time_of_day, user_profile, etc.)
            template: Prompt template to use
            
        Returns:
            List of LLMTrackRecommendation objects
            
        Example:
            >>> tracks = recommender.recommend(
            ...     mood_tag="relaxed but focused",
            ...     confidence=0.92,
            ...     n_tracks=3,
            ...     extra_context={"time_of_day": "afternoon", "activity": "studying"}
            ... )
        """
        # Build prompt
        prompt = self._build_prompt(
            mood_tag=mood_tag,
            confidence=confidence,
            n_tracks=n_tracks,
            extra_context=extra_context or {},
            template=template
        )
        
        logger.info(f"Requesting {n_tracks} tracks for mood: '{mood_tag}' (confidence: {confidence:.2%})")
        
        # Get recommendations from LLM or fallback
        if self.client is not None:
            try:
                tracks = self._query_llm(prompt, n_tracks)
            except Exception as e:
                logger.error(f"LLM query failed: {e}")
                if self.enable_fallback:
                    logger.info("Falling back to mock recommendations")
                    tracks = self._generate_fallback_recommendations(mood_tag, n_tracks)
                else:
                    raise
        else:
            if self.enable_fallback:
                tracks = self._generate_fallback_recommendations(mood_tag, n_tracks)
            else:
                raise RuntimeError("OpenAI client not available and fallback disabled")
        
        # Log to history
        self.recommendation_history.append({
            "timestamp": datetime.now().isoformat(),
            "mood_tag": mood_tag,
            "confidence": confidence,
            "n_tracks": n_tracks,
            "tracks": [t.to_dict() for t in tracks],
            "extra_context": extra_context
        })
        
        return tracks
    
    def _build_prompt(
        self,
        mood_tag: str,
        confidence: float,
        n_tracks: int,
        extra_context: Dict[str, Any],
        template: PromptTemplate
    ) -> str:
        """
        Build dynamic prompt for LLM based on mood and context.
        
        Args:
            mood_tag: Emotion/mood description
            confidence: Detection confidence
            n_tracks: Number of recommendations requested
            extra_context: Additional context
            template: Prompt template to use
            
        Returns:
            Formatted prompt string
        """
        # Extract context elements
        time_of_day = extra_context.get("time_of_day", self._get_time_of_day())
        activity = extra_context.get("activity", "listening")
        user_preferences = extra_context.get("preferences", "")
        history = extra_context.get("recent_tracks", [])
        
        # Confidence qualifier
        confidence_text = ""
        if confidence > 0.8:
            confidence_text = "with high confidence"
        elif confidence > 0.6:
            confidence_text = "with moderate confidence"
        else:
            confidence_text = "with some uncertainty"
        
        # Build prompt based on template
        if template == PromptTemplate.BASIC:
            prompt = f"""You are a music expert. Recommend {n_tracks} tracks for someone who is feeling {mood_tag}.

Respond with ONLY a numbered list in this exact format:
1. Artist Name - Song Title
2. Artist Name - Song Title
3. Artist Name - Song Title

No explanations, just the list."""
        
        elif template == PromptTemplate.DETAILED:
            prompt = f"""You are an expert music therapist and DJ. A user's emotional state has been detected via EEG brain signals.

**Detected Emotion:** {mood_tag} ({confidence_text}, {confidence:.0%} certainty)
**Time:** {time_of_day}
**Activity:** {activity}

Please recommend {n_tracks} specific, real music tracks that would:
1. Match and enhance their current emotional state
2. Be appropriate for {time_of_day} {activity}
3. Have the right energy level and mood

Format your response as a numbered list:
1. Artist Name - Song Title (brief reason)
2. Artist Name - Song Title (brief reason)
3. Artist Name - Song Title (brief reason)

Use real, well-known tracks that exist on Spotify."""
        
        elif template == PromptTemplate.CONTEXTUAL:
            recent_str = ""
            if history:
                recent_str = f"\n**Recently played:** {', '.join(history[:3])}"
            
            pref_str = ""
            if user_preferences:
                pref_str = f"\n**User preferences:** {user_preferences}"
            
            prompt = f"""As a music recommendation AI, analyze this user's current state and suggest optimal tracks.

**EEG-Detected Mood:** {mood_tag} {confidence_text} ({confidence:.0%})
**Context:** {time_of_day}, {activity}{recent_str}{pref_str}

Recommend {n_tracks} real, specific tracks available on Spotify that:
- Match the emotional valence and arousal level of "{mood_tag}"
- Fit the {time_of_day} timeframe
- Support the current activity: {activity}
- Avoid tracks too similar to recently played songs

Format:
1. Artist - Title | Reason: why this fits
2. Artist - Title | Reason: why this fits
3. Artist - Title | Reason: why this fits

Be specific and creative. Use actual songs."""
        
        elif template == PromptTemplate.THERAPEUTIC:
            prompt = f"""You are a music therapist using evidence-based music interventions.

**Client State (EEG-detected):** {mood_tag} (confidence: {confidence:.0%})
**Session Context:** {time_of_day} {activity}

Recommend {n_tracks} therapeutic music tracks that:
1. Provide emotional regulation for the detected state
2. Use iso-principle (match then guide emotional state)
3. Are available on Spotify streaming

Response format:
1. Artist - Song Title | Therapeutic goal: <goal>
2. Artist - Song Title | Therapeutic goal: <goal>
3. Artist - Song Title | Therapeutic goal: <goal>

Use evidence-based music therapy approaches."""
        
        else:
            # Default
            prompt = f"Recommend {n_tracks} songs for mood: {mood_tag}"
        
        return prompt
    
    def _query_llm(self, prompt: str, n_tracks: int) -> List[LLMTrackRecommendation]:
        """
        Query OpenAI API for recommendations.
        
        Args:
            prompt: Formatted prompt
            n_tracks: Expected number of tracks
            
        Returns:
            List of track recommendations
            
        Raises:
            Exception: If API call fails
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a music recommendation expert. Always provide real, specific song titles and artists that exist on Spotify. Format responses as requested."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                n=1
            )
            
            # Extract response
            content = response.choices[0].message.content.strip()
            logger.debug(f"LLM response:\n{content}")
            
            # Parse response into track recommendations
            tracks = self._parse_llm_response(content, n_tracks)
            
            if not tracks:
                logger.warning("No tracks parsed from LLM response, using fallback")
                raise ValueError("Failed to parse tracks from LLM response")
            
            return tracks
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def _parse_llm_response(self, content: str, expected_count: int) -> List[LLMTrackRecommendation]:
        """
        Parse LLM response to extract track recommendations.
        
        Supports multiple formats:
        - "1. Artist - Title"
        - "1. Artist - Title | Reason: ..."
        - "1. Artist - Title (reason)"
        
        Args:
            content: Raw LLM response text
            expected_count: Expected number of tracks
            
        Returns:
            List of parsed track recommendations
        """
        tracks = []
        
        # Split into lines
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        for line in lines:
            # Skip empty lines and headers
            if not line or line.startswith('#') or line.startswith('**'):
                continue
            
            # Try to match numbered list format: "1. Artist - Title | Reason: xyz" or "1. Artist - Title"
            # First try with pipe separator for reasoning
            match_with_pipe = re.match(r'^\d+[\.\)]\s*(.+?)\s*[-–—]\s*(.+?)\s*\|\s*(.+)$', line)
            if match_with_pipe:
                artist = match_with_pipe.group(1).strip()
                title = match_with_pipe.group(2).strip()
                reasoning = match_with_pipe.group(3).replace('Reason:', '').strip()
                
                tracks.append(LLMTrackRecommendation(
                    title=title,
                    artist=artist,
                    reasoning=reasoning,
                    confidence=0.8
                ))
                
                if len(tracks) >= expected_count:
                    break
                continue
            
            # Try basic format without reasoning
            match = re.match(r'^\d+[\.\)]\s*(.+?)\s*[-–—]\s*(.+)$', line)
            if match:
                artist = match.group(1).strip()
                title_and_more = match.group(2).strip()
                
                # Check for parentheses reasoning
                reasoning = ""
                if '(' in title_and_more:
                    title = title_and_more.split('(')[0].strip()
                    reasoning = title_and_more.split('(')[1].strip(')').strip()
                else:
                    title = title_and_more.strip()
                
                tracks.append(LLMTrackRecommendation(
                    title=title,
                    artist=artist,
                    reasoning=reasoning,
                    confidence=0.8 if reasoning else 0.7
                ))
                
                if len(tracks) >= expected_count:
                    break
        
        return tracks
    
    def _generate_fallback_recommendations(
        self,
        mood_tag: str,
        n_tracks: int
    ) -> List[LLMTrackRecommendation]:
        """
        Generate fallback recommendations when LLM unavailable.
        
        Uses a curated database of mood-to-track mappings.
        
        Args:
            mood_tag: Mood description
            n_tracks: Number of tracks to generate
            
        Returns:
            List of fallback recommendations
        """
        # Fallback database (mood -> tracks)
        fallback_db = {
            "happy": [
                ("Pharrell Williams", "Happy", "Upbeat, positive energy"),
                ("Katrina and the Waves", "Walking on Sunshine", "Feel-good classic"),
                ("Mark Ronson ft. Bruno Mars", "Uptown Funk", "High energy, fun"),
            ],
            "energetic": [
                ("Daft Punk", "One More Time", "Electronic energy boost"),
                ("The Black Keys", "Lonely Boy", "High tempo rock"),
                ("Avicii", "Wake Me Up", "Energizing EDM"),
            ],
            "calm": [
                ("Enya", "Only Time", "Soothing, peaceful"),
                ("Ludovico Einaudi", "Nuvole Bianche", "Gentle piano"),
                ("Sigur Rós", "Hoppípolla", "Ethereal, calming"),
            ],
            "relaxed": [
                ("Bon Iver", "Holocene", "Mellow, introspective"),
                ("Norah Jones", "Don't Know Why", "Smooth jazz"),
                ("Explosions in the Sky", "Your Hand in Mine", "Ambient post-rock"),
            ],
            "sad": [
                ("Adele", "Someone Like You", "Emotional ballad"),
                ("Jeff Buckley", "Hallelujah", "Melancholic beauty"),
                ("Radiohead", "Fake Plastic Trees", "Melancholy rock"),
            ],
            "focused": [
                ("Hans Zimmer", "Time", "Cinematic concentration"),
                ("Ólafur Arnalds", "Near Light", "Minimal focus music"),
                ("Max Richter", "On the Nature of Daylight", "Contemplative"),
            ],
            "stressed": [
                ("Weightless", "Marconi Union", "Scientifically calming"),
                ("Clair de Lune", "Claude Debussy", "Classical relaxation"),
                ("Electra", "Airstream", "Downtempo chill"),
            ],
        }
        
        # Find closest match in fallback database
        mood_lower = mood_tag.lower()
        best_match = "calm"  # default
        
        for key in fallback_db.keys():
            if key in mood_lower:
                best_match = key
                break
        
        # Get tracks for this mood
        track_list = fallback_db.get(best_match, fallback_db["calm"])
        
        # Convert to LLMTrackRecommendation objects
        tracks = []
        for i, (artist, title, reasoning) in enumerate(track_list[:n_tracks]):
            tracks.append(LLMTrackRecommendation(
                artist=artist,
                title=title,
                reasoning=reasoning,
                confidence=0.6  # Lower confidence for fallback
            ))
        
        # Fill up to n_tracks if needed
        while len(tracks) < n_tracks:
            tracks.append(tracks[len(tracks) % len(track_list)])
        
        logger.info(f"Generated {len(tracks)} fallback recommendations for mood: {mood_tag}")
        return tracks
    
    def _get_time_of_day(self) -> str:
        """Get current time of day as string."""
        hour = datetime.now().hour
        
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "night"
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get recommendation history."""
        return self.recommendation_history
    
    def clear_history(self) -> None:
        """Clear recommendation history."""
        self.recommendation_history.clear()
        logger.info("Recommendation history cleared")
    
    def __repr__(self) -> str:
        status = "connected" if self.client else "fallback mode"
        return f"LLMMusicRecommender(model={self.model}, status={status}, history={len(self.recommendation_history)})"


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_llm_recommender(
    api_key: Optional[str] = None,
    model: str = "gpt-4o",
    **kwargs
) -> LLMMusicRecommender:
    """
    Convenience function to create LLM recommender.
    
    Args:
        api_key: OpenAI API key
        model: Model to use
        **kwargs: Additional arguments for LLMMusicRecommender
        
    Returns:
        Configured LLMMusicRecommender instance
        
    Example:
        >>> recommender = create_llm_recommender(api_key="sk-...")
        >>> tracks = recommender.recommend("happy", confidence=0.9)
    """
    return LLMMusicRecommender(api_key=api_key, model=model, **kwargs)


def emotion_to_mood_tag(emotion: EmotionCategory, confidence: float = 1.0) -> str:
    """
    Convert EmotionCategory to descriptive mood tag for LLM prompts.
    
    Args:
        emotion: EmotionCategory enum
        confidence: Detection confidence
        
    Returns:
        Descriptive mood tag string
        
    Example:
        >>> tag = emotion_to_mood_tag(EmotionCategory.HAPPY, 0.85)
        >>> print(tag)  # "happy and energetic"
    """
    # Map emotions to descriptive tags
    mood_map = {
        EmotionCategory.HAPPY: "happy and energetic",
        EmotionCategory.CALM: "calm and peaceful",
        EmotionCategory.RELAXED: "relaxed and content",
        EmotionCategory.SAD: "sad and melancholic",
        EmotionCategory.ANGRY: "angry and intense",
        EmotionCategory.STRESSED: "stressed and anxious",
        EmotionCategory.EXCITED: "excited and enthusiastic",
        EmotionCategory.NEUTRAL: "neutral and balanced",
    }
    
    base_tag = mood_map.get(emotion, "neutral")
    
    # Modify based on confidence
    if confidence < 0.5:
        base_tag = f"somewhat {base_tag}"
    elif confidence > 0.9:
        base_tag = f"very {base_tag}"
    
    return base_tag
