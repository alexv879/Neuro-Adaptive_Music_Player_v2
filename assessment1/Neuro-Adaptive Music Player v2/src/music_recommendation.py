"""
Music Recommendation Engine for Neuro-Adaptive Music Player v2

This module provides intelligent music recommendation based on detected emotions
with support for multiple music streaming platforms (Spotify, YouTube, Deezer).

Features:
    - Emotion-to-music mapping with configurable mood profiles
    - Multi-platform support (Spotify, YouTube Music, Deezer, local files)
    - Intelligent playlist generation and shuffling
    - Playback control and state management
    - Song history and recommendation logging
    - Fallback strategies for API limitations

Author: Alexander V.
License: Proprietary
Version: 2.0.0
"""

import os
import json
import time
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

# Third-party imports
import numpy as np

# Optional music platform imports (graceful degradation)
try:
    import spotipy
    from spotipy.oauth2 import SpotifyOAuth, SpotifyClientCredentials
    SPOTIFY_AVAILABLE = True
except ImportError:
    SPOTIFY_AVAILABLE = False
    logging.warning("Spotify (spotipy) not installed. Spotify features disabled.")

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    logging.warning("Pygame not installed. Local music playback disabled.")

try:
    from pytube import YouTube
    YOUTUBE_AVAILABLE = True
except ImportError:
    YOUTUBE_AVAILABLE = False
    logging.warning("PyTube not installed. YouTube features disabled.")


# Configure logging
logger = logging.getLogger(__name__)


class EmotionCategory(Enum):
    """Emotion categories for music recommendation."""
    CALM = "calm"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    NEUTRAL = "neutral"
    EXCITED = "excited"
    RELAXED = "relaxed"
    STRESSED = "stressed"


class MusicPlatform(Enum):
    """Supported music streaming platforms."""
    SPOTIFY = "spotify"
    YOUTUBE = "youtube"
    DEEZER = "deezer"
    LOCAL = "local"
    NONE = "none"


@dataclass
class Track:
    """Represents a music track with metadata."""
    title: str
    artist: str
    uri: str  # Platform-specific URI (spotify:track:xxx, youtube URL, file path)
    platform: MusicPlatform
    emotion: EmotionCategory
    duration_ms: int = 0
    album: str = ""
    popularity: int = 0
    energy: float = 0.5
    valence: float = 0.5
    tempo: float = 120.0
    
    def __str__(self) -> str:
        return f"{self.artist} - {self.title}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert track to dictionary for logging."""
        return {
            "title": self.title,
            "artist": self.artist,
            "uri": self.uri,
            "platform": self.platform.value,
            "emotion": self.emotion.value,
            "duration_ms": self.duration_ms,
            "energy": self.energy,
            "valence": self.valence,
            "tempo": self.tempo
        }


@dataclass
class RecommendationHistory:
    """Tracks recommendation history for evaluation and personalization."""
    timestamp: datetime
    emotion_detected: EmotionCategory
    emotion_confidence: float
    track_played: Track
    user_feedback: Optional[str] = None  # "liked", "disliked", "skipped", None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert history entry to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "emotion": self.emotion_detected.value,
            "confidence": self.emotion_confidence,
            "track": self.track_played.to_dict(),
            "feedback": self.user_feedback
        }


class MusicRecommendationEngine:
    """
    Intelligent music recommendation engine with multi-platform support.
    
    This class handles emotion-based music selection, playlist management,
    and playback control across different music streaming platforms.
    
    Attributes:
        platform (MusicPlatform): Primary music platform to use
        emotion_to_genre (Dict): Mapping from emotions to music genres/moods
        history (List[RecommendationHistory]): Recommendation history
        current_track (Optional[Track]): Currently playing track
        
    Example:
        >>> engine = MusicRecommendationEngine(platform=MusicPlatform.SPOTIFY)
        >>> engine.authenticate_spotify(client_id="xxx", client_secret="yyy")
        >>> track = engine.recommend(emotion="happy", confidence=0.85)
        >>> engine.play(track)
    """
    
    def __init__(
        self,
        platform: MusicPlatform = MusicPlatform.SPOTIFY,
        config_path: Optional[str] = None,
        cache_dir: str = "music_cache"
    ):
        """
        Initialize the music recommendation engine.
        
        Args:
            platform: Primary music platform (Spotify, YouTube, local, etc.)
            config_path: Path to configuration file with API credentials
            cache_dir: Directory for caching playlist data
        """
        self.platform = platform
        self.config_path = config_path
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Recommendation history
        self.history: List[RecommendationHistory] = []
        self.current_track: Optional[Track] = None
        
        # Platform clients (initialized on demand)
        self.spotify_client: Optional[Any] = None
        self.pygame_mixer: Optional[Any] = None
        
        # Emotion-to-music mapping (valence-arousal model)
        self.emotion_profiles = self._initialize_emotion_profiles()
        
        # Load configuration if provided
        if config_path and os.path.exists(config_path):
            self._load_config(config_path)
        
        logger.info(f"MusicRecommendationEngine initialized with platform: {platform.value}")
    
    def _initialize_emotion_profiles(self) -> Dict[EmotionCategory, Dict[str, Any]]:
        """
        Initialize emotion profiles with Spotify audio features and genres.
        
        Based on Russell's Circumplex Model of Affect (valence-arousal).
        
        Returns:
            Dictionary mapping emotions to music characteristics
        """
        return {
            EmotionCategory.HAPPY: {
                "valence_range": (0.6, 1.0),  # High positive valence
                "energy_range": (0.6, 1.0),   # High arousal
                "tempo_range": (110, 140),     # Upbeat tempo
                "genres": ["pop", "dance", "funk", "disco", "indie"],
                "spotify_seeds": ["happy", "party", "upbeat", "summer"],
                "search_terms": ["happy", "uplifting", "feel good", "cheerful"]
            },
            EmotionCategory.CALM: {
                "valence_range": (0.4, 0.7),
                "energy_range": (0.1, 0.4),
                "tempo_range": (60, 90),
                "genres": ["ambient", "classical", "acoustic", "chill"],
                "spotify_seeds": ["calm", "peaceful", "meditation", "relaxing"],
                "search_terms": ["calm", "peaceful", "serene", "tranquil"]
            },
            EmotionCategory.SAD: {
                "valence_range": (0.0, 0.4),
                "energy_range": (0.1, 0.5),
                "tempo_range": (60, 100),
                "genres": ["sad", "blues", "ballad", "acoustic"],
                "spotify_seeds": ["sad", "melancholy", "emotional", "heartbreak"],
                "search_terms": ["sad", "melancholic", "emotional", "somber"]
            },
            EmotionCategory.ANGRY: {
                "valence_range": (0.0, 0.4),
                "energy_range": (0.7, 1.0),
                "tempo_range": (120, 180),
                "genres": ["metal", "rock", "punk", "hardcore"],
                "spotify_seeds": ["angry", "aggressive", "intense", "rage"],
                "search_terms": ["angry", "aggressive", "intense", "powerful"]
            },
            EmotionCategory.EXCITED: {
                "valence_range": (0.6, 1.0),
                "energy_range": (0.7, 1.0),
                "tempo_range": (120, 160),
                "genres": ["electronic", "edm", "dance", "techno"],
                "spotify_seeds": ["workout", "energy", "power", "motivation"],
                "search_terms": ["energetic", "exciting", "dynamic", "powerful"]
            },
            EmotionCategory.RELAXED: {
                "valence_range": (0.5, 0.8),
                "energy_range": (0.2, 0.5),
                "tempo_range": (70, 100),
                "genres": ["jazz", "lounge", "chillout", "downtempo"],
                "spotify_seeds": ["chill", "easy listening", "smooth", "lounge"],
                "search_terms": ["relaxing", "smooth", "mellow", "easy"]
            },
            EmotionCategory.NEUTRAL: {
                "valence_range": (0.4, 0.6),
                "energy_range": (0.4, 0.6),
                "tempo_range": (90, 120),
                "genres": ["pop", "indie", "folk", "alternative"],
                "spotify_seeds": ["chill", "indie", "alternative"],
                "search_terms": ["moderate", "balanced", "neutral"]
            },
            EmotionCategory.STRESSED: {
                "valence_range": (0.2, 0.5),
                "energy_range": (0.6, 0.9),
                "tempo_range": (100, 140),
                "genres": ["ambient", "meditation", "classical", "nature sounds"],
                "spotify_seeds": ["stress relief", "calming", "meditation"],
                "search_terms": ["stress relief", "calming", "soothing"]
            }
        }
    
    def _load_config(self, config_path: str) -> None:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Load Spotify credentials
            if 'spotify' in config:
                spotify_config = config['spotify']
                self.authenticate_spotify(
                    client_id=spotify_config.get('client_id'),
                    client_secret=spotify_config.get('client_secret'),
                    redirect_uri=spotify_config.get('redirect_uri', 'http://localhost:8888/callback')
                )
            
            # Load custom emotion profiles (optional)
            if 'emotion_profiles' in config:
                self.emotion_profiles.update(config['emotion_profiles'])
            
            logger.info(f"Configuration loaded from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
    
    # ==================== AUTHENTICATION ====================
    
    def authenticate_spotify(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        redirect_uri: str = "http://localhost:8888/callback",
        scope: str = "user-modify-playback-state user-read-playback-state playlist-read-private"
    ) -> bool:
        """
        Authenticate with Spotify API.
        
        Args:
            client_id: Spotify application client ID
            client_secret: Spotify application client secret
            redirect_uri: OAuth redirect URI
            scope: Spotify API scopes
            
        Returns:
            True if authentication successful, False otherwise
        """
        if not SPOTIFY_AVAILABLE:
            logger.error("Spotify (spotipy) not installed. Install with: pip install spotipy")
            return False
        
        try:
            # Try environment variables first
            client_id = client_id or os.getenv('SPOTIFY_CLIENT_ID')
            client_secret = client_secret or os.getenv('SPOTIFY_CLIENT_SECRET')
            
            if not client_id or not client_secret:
                logger.error("Spotify credentials not provided. Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET.")
                return False
            
            # Use OAuth for user-specific operations (playback control)
            auth_manager = SpotifyOAuth(
                client_id=client_id,
                client_secret=client_secret,
                redirect_uri=redirect_uri,
                scope=scope,
                cache_path=str(self.cache_dir / ".spotify_cache")
            )
            
            self.spotify_client = spotipy.Spotify(auth_manager=auth_manager)
            
            # Test authentication
            user = self.spotify_client.current_user()
            logger.info(f"Authenticated with Spotify as: {user['display_name']}")
            return True
            
        except Exception as e:
            logger.error(f"Spotify authentication failed: {e}")
            self.spotify_client = None
            return False
    
    def authenticate_spotify_simple(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None
    ) -> bool:
        """
        Authenticate with Spotify using client credentials (no user login).
        
        This method is suitable for search and recommendations but not playback control.
        
        Args:
            client_id: Spotify application client ID
            client_secret: Spotify application client secret
            
        Returns:
            True if authentication successful, False otherwise
        """
        if not SPOTIFY_AVAILABLE:
            logger.error("Spotify (spotipy) not installed.")
            return False
        
        try:
            client_id = client_id or os.getenv('SPOTIFY_CLIENT_ID')
            client_secret = client_secret or os.getenv('SPOTIFY_CLIENT_SECRET')
            
            if not client_id or not client_secret:
                logger.error("Spotify credentials not provided.")
                return False
            
            auth_manager = SpotifyClientCredentials(
                client_id=client_id,
                client_secret=client_secret
            )
            
            self.spotify_client = spotipy.Spotify(auth_manager=auth_manager)
            logger.info("Authenticated with Spotify (client credentials mode)")
            return True
            
        except Exception as e:
            logger.error(f"Spotify authentication failed: {e}")
            return False
    
    # ==================== RECOMMENDATION ====================
    
    def recommend(
        self,
        emotion: Union[str, EmotionCategory],
        confidence: float = 1.0,
        n_tracks: int = 1,
        diversity: float = 0.3
    ) -> Union[Track, List[Track]]:
        """
        Recommend music based on detected emotion.
        
        Args:
            emotion: Detected emotion (string or EmotionCategory)
            confidence: Confidence of emotion detection (0-1)
            n_tracks: Number of tracks to recommend
            diversity: How diverse recommendations should be (0=safe, 1=adventurous)
            
        Returns:
            Single Track if n_tracks=1, else list of Tracks
            
        Example:
            >>> track = engine.recommend("happy", confidence=0.85)
            >>> tracks = engine.recommend(EmotionCategory.CALM, n_tracks=5)
        """
        # Convert string to EmotionCategory
        if isinstance(emotion, str):
            try:
                emotion = EmotionCategory(emotion.lower())
            except ValueError:
                logger.warning(f"Unknown emotion '{emotion}', defaulting to NEUTRAL")
                emotion = EmotionCategory.NEUTRAL
        
        logger.info(f"Recommending music for emotion: {emotion.value} (confidence: {confidence:.2f})")
        
        # Select recommendation strategy based on platform
        if self.platform == MusicPlatform.SPOTIFY and self.spotify_client:
            tracks = self._recommend_spotify(emotion, n_tracks, diversity)
        elif self.platform == MusicPlatform.YOUTUBE:
            tracks = self._recommend_youtube(emotion, n_tracks)
        elif self.platform == MusicPlatform.LOCAL:
            tracks = self._recommend_local(emotion, n_tracks)
        else:
            # Fallback to mock recommendations
            logger.warning(f"Platform {self.platform.value} not available, using mock recommendations")
            tracks = self._recommend_mock(emotion, n_tracks)
        
        # Return single track or list
        if n_tracks == 1:
            return tracks[0] if tracks else self._get_fallback_track(emotion)
        return tracks
    
    def _recommend_spotify(
        self,
        emotion: EmotionCategory,
        n_tracks: int,
        diversity: float
    ) -> List[Track]:
        """Recommend tracks from Spotify based on emotion profile."""
        if not self.spotify_client:
            logger.error("Spotify client not authenticated")
            return []
        
        profile = self.emotion_profiles[emotion]
        tracks = []
        
        try:
            # Strategy 1: Search by mood/genre
            search_terms = profile["search_terms"]
            genres = profile["genres"]
            
            # Combine search terms and genres
            query_options = [
                f"genre:{random.choice(genres)}",
                f"{random.choice(search_terms)} {random.choice(genres)}",
                f"mood:{random.choice(profile['spotify_seeds'])}"
            ]
            
            for _ in range(min(n_tracks, 3)):
                query = random.choice(query_options)
                
                results = self.spotify_client.search(
                    q=query,
                    type='track',
                    limit=20
                )
                
                if results['tracks']['items']:
                    # Filter by audio features
                    track_data = random.choice(results['tracks']['items'])
                    
                    # Get audio features
                    audio_features = self.spotify_client.audio_features([track_data['id']])[0]
                    
                    if audio_features:
                        valence = audio_features['valence']
                        energy = audio_features['energy']
                        tempo = audio_features['tempo']
                        
                        # Check if track matches emotion profile
                        valence_match = profile['valence_range'][0] <= valence <= profile['valence_range'][1]
                        energy_match = profile['energy_range'][0] <= energy <= profile['energy_range'][1]
                        
                        # Allow some diversity
                        if valence_match or energy_match or diversity > 0.5:
                            track = Track(
                                title=track_data['name'],
                                artist=track_data['artists'][0]['name'],
                                uri=track_data['uri'],
                                platform=MusicPlatform.SPOTIFY,
                                emotion=emotion,
                                duration_ms=track_data['duration_ms'],
                                album=track_data['album']['name'],
                                popularity=track_data['popularity'],
                                energy=energy,
                                valence=valence,
                                tempo=tempo
                            )
                            tracks.append(track)
                            
                            if len(tracks) >= n_tracks:
                                break
            
            # Strategy 2: Use Spotify recommendations API (if we need more tracks)
            if len(tracks) < n_tracks:
                seed_genres = random.sample(genres, min(3, len(genres)))
                
                recommendations = self.spotify_client.recommendations(
                    seed_genres=seed_genres,
                    limit=n_tracks - len(tracks),
                    target_valence=np.mean(profile['valence_range']),
                    target_energy=np.mean(profile['energy_range']),
                    target_tempo=np.mean(profile['tempo_range'])
                )
                
                for track_data in recommendations['tracks']:
                    track = Track(
                        title=track_data['name'],
                        artist=track_data['artists'][0]['name'],
                        uri=track_data['uri'],
                        platform=MusicPlatform.SPOTIFY,
                        emotion=emotion,
                        duration_ms=track_data['duration_ms'],
                        album=track_data['album']['name'],
                        popularity=track_data['popularity']
                    )
                    tracks.append(track)
            
            logger.info(f"Found {len(tracks)} Spotify tracks for {emotion.value}")
            return tracks
            
        except Exception as e:
            logger.error(f"Spotify recommendation failed: {e}")
            return []
    
    def _recommend_youtube(self, emotion: EmotionCategory, n_tracks: int) -> List[Track]:
        """Recommend tracks from YouTube (placeholder for future implementation)."""
        logger.warning("YouTube recommendations not yet implemented")
        return self._recommend_mock(emotion, n_tracks)
    
    def _recommend_local(self, emotion: EmotionCategory, n_tracks: int) -> List[Track]:
        """Recommend tracks from local music library."""
        music_dir = Path("music") / emotion.value
        
        if not music_dir.exists():
            logger.warning(f"Local music directory not found: {music_dir}")
            return []
        
        # Find all audio files
        audio_extensions = ['.mp3', '.wav', '.ogg', '.flac', '.m4a']
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(music_dir.glob(f"*{ext}"))
        
        if not audio_files:
            logger.warning(f"No audio files found in {music_dir}")
            return []
        
        # Sample random tracks
        selected_files = random.sample(audio_files, min(n_tracks, len(audio_files)))
        
        tracks = []
        for file_path in selected_files:
            # Parse filename for artist-title (format: "Artist - Title.mp3")
            filename = file_path.stem
            if ' - ' in filename:
                artist, title = filename.split(' - ', 1)
            else:
                artist = "Unknown"
                title = filename
            
            track = Track(
                title=title,
                artist=artist,
                uri=str(file_path.absolute()),
                platform=MusicPlatform.LOCAL,
                emotion=emotion
            )
            tracks.append(track)
        
        logger.info(f"Found {len(tracks)} local tracks for {emotion.value}")
        return tracks
    
    def _recommend_mock(self, emotion: EmotionCategory, n_tracks: int) -> List[Track]:
        """Generate mock recommendations for testing."""
        profile = self.emotion_profiles[emotion]
        tracks = []
        
        mock_data = {
            EmotionCategory.HAPPY: [
                ("Pharrell Williams", "Happy"),
                ("Katrina and The Waves", "Walking on Sunshine"),
                ("Bobby McFerrin", "Don't Worry Be Happy")
            ],
            EmotionCategory.CALM: [
                ("Enya", "Only Time"),
                ("Ludovico Einaudi", "Nuvole Bianche"),
                ("Brian Eno", "An Ending (Ascent)")
            ],
            EmotionCategory.SAD: [
                ("Adele", "Someone Like You"),
                ("Sam Smith", "Stay With Me"),
                ("Billie Eilish", "when the party's over")
            ],
            EmotionCategory.ANGRY: [
                ("Rage Against The Machine", "Killing In The Name"),
                ("Metallica", "Enter Sandman"),
                ("Linkin Park", "In The End")
            ]
        }
        
        song_list = mock_data.get(emotion, [("Artist", "Song")])
        
        for i in range(min(n_tracks, len(song_list))):
            artist, title = song_list[i % len(song_list)]
            track = Track(
                title=title,
                artist=artist,
                uri=f"mock://track/{emotion.value}/{i}",
                platform=MusicPlatform.NONE,
                emotion=emotion
            )
            tracks.append(track)
        
        return tracks
    
    def _get_fallback_track(self, emotion: EmotionCategory) -> Track:
        """Get a fallback track when recommendations fail."""
        return Track(
            title="Fallback Song",
            artist="Unknown Artist",
            uri="fallback://track",
            platform=MusicPlatform.NONE,
            emotion=emotion
        )
    
    # ==================== PLAYBACK CONTROL ====================
    
    def play(self, track: Track, log_history: bool = True) -> bool:
        """
        Play a track on the appropriate platform.
        
        Args:
            track: Track to play
            log_history: Whether to log this playback in history
            
        Returns:
            True if playback started successfully, False otherwise
        """
        logger.info(f"Playing: {track}")
        
        success = False
        
        if track.platform == MusicPlatform.SPOTIFY:
            success = self._play_spotify(track)
        elif track.platform == MusicPlatform.LOCAL:
            success = self._play_local(track)
        elif track.platform == MusicPlatform.YOUTUBE:
            success = self._play_youtube(track)
        else:
            logger.warning(f"Cannot play track from platform: {track.platform.value}")
            success = False
        
        if success:
            self.current_track = track
            
            if log_history:
                history_entry = RecommendationHistory(
                    timestamp=datetime.now(),
                    emotion_detected=track.emotion,
                    emotion_confidence=1.0,
                    track_played=track
                )
                self.history.append(history_entry)
        
        return success
    
    def _play_spotify(self, track: Track) -> bool:
        """Play track on Spotify."""
        if not self.spotify_client:
            logger.error("Spotify client not authenticated")
            return False
        
        try:
            # Get available devices
            devices = self.spotify_client.devices()
            
            if not devices['devices']:
                logger.error("No active Spotify devices found. Open Spotify on a device first.")
                return False
            
            # Start playback on first available device
            device_id = devices['devices'][0]['id']
            self.spotify_client.start_playback(device_id=device_id, uris=[track.uri])
            
            logger.info(f"Started Spotify playback: {track}")
            return True
            
        except Exception as e:
            logger.error(f"Spotify playback failed: {e}")
            return False
    
    def _play_local(self, track: Track) -> bool:
        """Play local audio file using pygame."""
        if not PYGAME_AVAILABLE:
            logger.error("Pygame not installed. Cannot play local files.")
            return False
        
        try:
            # Initialize pygame mixer if not already initialized
            if not self.pygame_mixer:
                pygame.mixer.init()
                self.pygame_mixer = pygame.mixer.music
            
            # Load and play
            self.pygame_mixer.load(track.uri)
            self.pygame_mixer.play()
            
            logger.info(f"Started local playback: {track}")
            return True
            
        except Exception as e:
            logger.error(f"Local playback failed: {e}")
            return False
    
    def _play_youtube(self, track: Track) -> bool:
        """Play YouTube video (placeholder)."""
        logger.warning("YouTube playback not yet implemented")
        return False
    
    def pause(self) -> bool:
        """Pause current playback."""
        if not self.current_track:
            logger.warning("No track currently playing")
            return False
        
        try:
            if self.current_track.platform == MusicPlatform.SPOTIFY and self.spotify_client:
                self.spotify_client.pause_playback()
                return True
            elif self.current_track.platform == MusicPlatform.LOCAL and self.pygame_mixer:
                self.pygame_mixer.pause()
                return True
        except Exception as e:
            logger.error(f"Pause failed: {e}")
        
        return False
    
    def resume(self) -> bool:
        """Resume paused playback."""
        if not self.current_track:
            logger.warning("No track to resume")
            return False
        
        try:
            if self.current_track.platform == MusicPlatform.SPOTIFY and self.spotify_client:
                self.spotify_client.start_playback()
                return True
            elif self.current_track.platform == MusicPlatform.LOCAL and self.pygame_mixer:
                self.pygame_mixer.unpause()
                return True
        except Exception as e:
            logger.error(f"Resume failed: {e}")
        
        return False
    
    def skip(self) -> bool:
        """Skip to next track."""
        if not self.current_track:
            return False
        
        # Log skip as feedback
        if self.history and self.history[-1].track_played == self.current_track:
            self.history[-1].user_feedback = "skipped"
        
        try:
            if self.current_track.platform == MusicPlatform.SPOTIFY and self.spotify_client:
                self.spotify_client.next_track()
                return True
            elif self.current_track.platform == MusicPlatform.LOCAL and self.pygame_mixer:
                self.pygame_mixer.stop()
                return True
        except Exception as e:
            logger.error(f"Skip failed: {e}")
        
        return False
    
    # ==================== HISTORY & LOGGING ====================
    
    def save_history(self, filepath: Optional[str] = None) -> None:
        """Save recommendation history to JSON file."""
        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.cache_dir / f"history_{timestamp}.json"
        
        history_data = [entry.to_dict() for entry in self.history]
        
        with open(filepath, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        logger.info(f"Saved {len(self.history)} history entries to {filepath}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about recommendation history."""
        if not self.history:
            return {"total_tracks": 0}
        
        emotion_counts = {}
        platform_counts = {}
        feedback_counts = {"liked": 0, "disliked": 0, "skipped": 0, "none": 0}
        
        for entry in self.history:
            # Count emotions
            emotion = entry.emotion_detected.value
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            # Count platforms
            platform = entry.track_played.platform.value
            platform_counts[platform] = platform_counts.get(platform, 0) + 1
            
            # Count feedback
            feedback = entry.user_feedback or "none"
            feedback_counts[feedback] = feedback_counts.get(feedback, 0) + 1
        
        return {
            "total_tracks": len(self.history),
            "emotions": emotion_counts,
            "platforms": platform_counts,
            "feedback": feedback_counts,
            "average_confidence": np.mean([e.emotion_confidence for e in self.history])
        }
    
    def __repr__(self) -> str:
        return f"MusicRecommendationEngine(platform={self.platform.value}, tracks_played={len(self.history)})"


# ==================== CONVENIENCE FUNCTIONS ====================

def create_recommendation_engine(
    platform: str = "spotify",
    spotify_client_id: Optional[str] = None,
    spotify_client_secret: Optional[str] = None
) -> MusicRecommendationEngine:
    """
    Convenience function to create and configure a recommendation engine.
    
    Args:
        platform: Music platform ("spotify", "youtube", "local", "none")
        spotify_client_id: Spotify API client ID (or set SPOTIFY_CLIENT_ID env var)
        spotify_client_secret: Spotify API client secret (or set SPOTIFY_CLIENT_SECRET env var)
        
    Returns:
        Configured MusicRecommendationEngine instance
        
    Example:
        >>> engine = create_recommendation_engine("spotify")
        >>> track = engine.recommend("happy")
        >>> engine.play(track)
    """
    platform_enum = MusicPlatform(platform.lower())
    engine = MusicRecommendationEngine(platform=platform_enum)
    
    if platform_enum == MusicPlatform.SPOTIFY:
        # Try to authenticate with Spotify
        engine.authenticate_spotify_simple(
            client_id=spotify_client_id,
            client_secret=spotify_client_secret
        )
    
    return engine


# ==================== SELF-TEST ====================

if __name__ == "__main__":
    """Self-test and demonstration of the music recommendation engine."""
    
    print("=" * 60)
    print("Music Recommendation Engine - Self Test")
    print("=" * 60)
    
    # Test 1: Create engine with mock platform
    print("\n[Test 1] Creating engine with mock platform...")
    engine = MusicRecommendationEngine(platform=MusicPlatform.NONE)
    print(f"✓ Engine created: {engine}")
    
    # Test 2: Test all emotion categories
    print("\n[Test 2] Testing recommendations for all emotions...")
    for emotion in EmotionCategory:
        tracks = engine.recommend(emotion, n_tracks=2)
        print(f"  {emotion.value:12} → {len(tracks)} tracks: {tracks[0]}")
    
    # Test 3: Test recommendation with confidence
    print("\n[Test 3] Testing recommendations with varying confidence...")
    for conf in [0.5, 0.75, 0.95]:
        track = engine.recommend("happy", confidence=conf)
        print(f"  Confidence {conf:.2f} → {track}")
    
    # Test 4: Test mock playback
    print("\n[Test 4] Testing mock playback...")
    track = engine.recommend("calm")
    success = engine.play(track)
    print(f"  Playback {'successful' if success else 'failed'}: {track}")
    
    # Test 5: Test history
    print("\n[Test 5] Testing history tracking...")
    for emotion in ["happy", "sad", "calm"]:
        track = engine.recommend(emotion)
        engine.play(track)
    
    stats = engine.get_statistics()
    print(f"  Total tracks played: {stats['total_tracks']}")
    print(f"  Emotions: {stats['emotions']}")
    
    # Test 6: Save history
    print("\n[Test 6] Saving history...")
    engine.save_history()
    print(f"  ✓ History saved to music_cache/")
    
    # Test 7: Test Spotify (if credentials available)
    print("\n[Test 7] Testing Spotify integration...")
    if os.getenv('SPOTIFY_CLIENT_ID'):
        spotify_engine = create_recommendation_engine("spotify")
        if spotify_engine.spotify_client:
            print("  ✓ Spotify authentication successful")
            track = spotify_engine.recommend("happy")
            print(f"  Recommended: {track}")
        else:
            print("  ✗ Spotify authentication failed")
    else:
        print("  ⊘ Skipped (SPOTIFY_CLIENT_ID not set)")
    
    print("\n" + "=" * 60)
    print("Self-test complete! ✓")
    print("=" * 60)
    print("\nTo use Spotify:")
    print("  1. Create app at https://developer.spotify.com/dashboard")
    print("  2. Set environment variables:")
    print("     export SPOTIFY_CLIENT_ID='your_client_id'")
    print("     export SPOTIFY_CLIENT_SECRET='your_client_secret'")
    print("  3. Run: python -m src.music_recommendation")
