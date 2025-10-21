# Configuration settings for Neuro-Adaptive Music Player
# This file centralizes all app settings for easy adjustment without code changes.

# EEG settings
EEG_SIMULATED = True  # Use simulated data for demo (Week 1: Signal fidelity - real devices would stream live data)
EEG_CHANNEL = 0  # Selected channel index (e.g., single channel from EDF files in assessment)
WINDOW_SIZE = 5  # seconds for rolling window (real-time processing window for continuous analysis)
FS = 256  # Sampling rate Hz (matches clinical EEG standards from Week 1: Nyquist and sampling theory)

# Signal processing
BANDPASS_LOW = 1  # Hz (low cutoff for EEG band isolation, Week 3: Convolution theory for filtering)
BANDPASS_HIGH = 30  # Hz (high cutoff, removes noise above beta band)
BUTTER_ORDER = 4  # Filter order for Butterworth (balances sharpness and ripple)

# Brain state thresholds (example values, tunable based on user/research)
FOCUS_THRESHOLD = 1.5  # beta/alpha ratio (high for focus, Week 2: Frequency bands)
RELAX_THRESHOLD = 0.8  # alpha/theta ratio (high for relax)
FATIGUE_THRESHOLD = 0.5  # overall power drop (low for fatigue)
HAPPY_THRESHOLD = 2.0  # alpha/beta ratio (high for happiness, based on positive mood research)
SAD_THRESHOLD = 0.3  # theta/alpha ratio (high for sadness, indicative of low mood)

# Music settings
MUSIC_FOLDER = "music/"  # Relative path to music files (for local playback)
USE_LOCAL = True  # If True, play local files; if False, open Amazon Music search (flexible for different setups)
AMAZON_MUSIC_URL = "https://www.amazon.com/s?k={}&i=digital-music"  # Search URL template (web integration for streaming)

# OpenAI settings (for AI-driven recommendations)
OPENAI_API_KEY = "your-openai-api-key-here"  # Replace with your key (secure API access)
MODEL = "gpt-3.5-turbo"  # Or gpt-4 (language model for song suggestions, agentic AI)

# Logging
LOG_FILE = "session_log.csv"  # CSV for session data (Week 2: Scientific coding - reproducible logging for analysis)