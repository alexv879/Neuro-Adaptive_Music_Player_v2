# Music player module with AI-driven recommendations and playback
# Integrates OpenAI for agentic song suggestions based on brain states (extends Week 4: Adaptive systems)

import pygame  # For local audio playback
import os  # For file system operations
import openai  # For AI song recommendations
import webbrowser  # For opening Amazon Music searches
import time  # For delays in transitions
from config import OPENAI_API_KEY, MODEL, MUSIC_FOLDER, USE_LOCAL, AMAZON_MUSIC_URL  # Import settings
import time  # For sleep during fade out

openai.api_key = OPENAI_API_KEY  # Set API key for OpenAI calls

class MusicPlayer:
    # Initializes the music player based on local or streaming mode
    def __init__(self):
        if USE_LOCAL:  # If using local files, initialize pygame mixer
            pygame.mixer.init()  # Set up audio playback
        self.music_folder = MUSIC_FOLDER  # Path to local music
        self.current_song = None  # Track currently playing song
        self.song_queue = []  # Queue of 3 recommended songs
        self.queue_index = -1  # Current index in queue (-1 means no queue)

    # Uses OpenAI to recommend 3 songs based on brain state (agentic AI for multiple options)
    def recommend_songs(self, state):
        prompt = f"Recommend 3 song titles suitable for a '{state}' brain state. List them as '1. Song Name by Artist', '2. ...', '3. ...'. Focus on variety but aligned with the state."
        try:
            response = openai.ChatCompletion.create(  # Call OpenAI API
                model=MODEL,  # Use specified model
                messages=[{"role": "user", "content": prompt}],  # Send prompt
                max_tokens=150  # Allow for 3 songs
            )
            songs_text = response.choices[0].message.content.strip()  # Extract response
            # Parse into list: assume format "1. Song by Artist\n2. ..."
            songs = []
            for line in songs_text.split('\n'):
                if '. ' in line:
                    song = line.split('. ', 1)[1].strip()
                    songs.append(song)
            return songs[:3] if len(songs) >= 3 else songs + ["Default Song by Unknown"] * (3 - len(songs))  # Ensure 3
        except Exception as e:
            print(f"OpenAI error: {e}")  # Handle API errors
            return ["Default Song 1 by Unknown", "Default Song 2 by Unknown", "Default Song 3 by Unknown"]  # Fallback

    # Searches local music folder for a matching MP3 file
    def find_local_song(self, song_name):
        # Simple search: check if song_name matches a file in music/
        for file in os.listdir(self.music_folder):  # List files in folder
            if song_name.lower() in file.lower() and file.endswith('.mp3'):  # Case-insensitive match
                return os.path.join(self.music_folder, file)  # Return full path
        return None  # No match found

    # Main method: Recommends 3 songs and starts playing the first based on state
    def play_song(self, state):
        self.song_queue = self.recommend_songs(state)  # Get 3 AI recommendations
        self.queue_index = 0  # Start with first song
        self._play_current_from_queue()  # Play the first song

    # Stops current playback
    def stop(self):
        if USE_LOCAL:
            pygame.mixer.music.stop()  # Stop pygame playback

    # Fades out current song for smooth transitions
    def fade_out(self, ms=1000):
        if USE_LOCAL:
            pygame.mixer.music.fadeout(ms)  # Fade out over ms milliseconds

    # Returns the current song for logging/display
    def get_current_song(self):
        return self.current_song or "None"

    # Internal method to play the current song from the queue
    def _play_current_from_queue(self):
        if self.queue_index < len(self.song_queue):
            song = self.song_queue[self.queue_index]
            if USE_LOCAL:  # Local playback mode
                local_path = self.find_local_song(song)  # Search for file
                if local_path:
                    pygame.mixer.music.load(local_path)  # Load MP3
                    pygame.mixer.music.play()  # Start playback
                    self.current_song = song  # Update current song
                else:
                    print(f"Song '{song}' not found locally. Skipping to next.")  # Notify user
                    self.skip_to_next()  # Auto-skip if not found
            else:  # Amazon Music search mode
                # Open Amazon Music search in browser
                search_query = song.replace(" ", "+")  # Format for URL
                url = AMAZON_MUSIC_URL.format(search_query)  # Build search URL
                webbrowser.open(url)  # Open in default browser
                self.current_song = f"Searching Amazon: {song}"  # Update status
                print(f"Opened Amazon Music search for: {song}")  # Console feedback

    # Skips to the next song in the queue; if none left, re-recommends for the state
    def skip_to_next(self, state_if_needed=None):
        self.fade_out(500)  # Fade out current
        time.sleep(0.5)
        self.queue_index += 1
        if self.queue_index < len(self.song_queue):
            self._play_current_from_queue()
        else:
            print("All songs skipped. Re-recommending...")
            if state_if_needed:
                self.play_song(state_if_needed)  # Re-start with new recommendations
            else:
                self.current_song = "None"