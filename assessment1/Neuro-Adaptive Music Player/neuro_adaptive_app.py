from eeg_signal_processor import SignalProcessor  # Imports EEG signal processing module
from ai_music_controller import MusicPlayer  # Imports AI music control with recommendations
from session_data_logger import Logger  # Imports logging for session data
import time  # For timing delays
import signal  # For handling interrupts
import sys  # For system exit

# Signal handler for graceful shutdown on Ctrl+C
def signal_handler(sig, frame):
    print("Stopping session...")
    player.stop()  # Stop any playing music
    sys.exit(0)  # Exit cleanly

signal.signal(signal.SIGINT, signal_handler)  # Register handler for SIGINT

# Initialize core components
processor = SignalProcessor()  # For EEG signal processing and state classification
player = MusicPlayer()  # For AI music recommendations and playback
logger = Logger()  # For logging features, states, and songs

# Main session loop: Continuously analyzes EEG and adapts music
def run_session():
    from eeg_data_simulator import EEGSimulator  # Import simulator for demo data
    eeg = EEGSimulator()  # Initialize simulated EEG stream
    last_state = None  # Track previous state to detect changes
    feedback_counter = 0  # Counter for periodic feedback

    print("Starting EEG analysis and AI music adaptation. Press Ctrl+C to stop.")
    try:
        while True:  # Infinite loop for real-time operation
            signal_data, sim_state = eeg.get_data_stream()  # Get 5-second window of EEG data (Week 1: Signal streaming)
            state, features = processor.process_window(signal_data)  # Process and classify brain state (Week 2-4: Filtering, PSD, artifact detection)

            if state != last_state:  # If state changed, adapt music
                print(f"State changed to: {state}")
                player.fade_out(500)  # Smooth transition (fade out current song)
                time.sleep(0.5)  # Brief pause
                player.play_song(state)  # AI recommends 3 songs and plays first
                last_state = state  # Update last state

                # Immediate feedback for the first song
                try:
                    feedback = input(f"Playing: {player.get_current_song()}. Like it? (y/n/skip): ").strip().lower()
                    if feedback in ['n', 'skip']:
                        player.skip_to_next(state)  # Skip to next in queue or re-recommend
                        logger.log_feedback(player.get_current_song(), feedback)
                    elif feedback == 'y':
                        logger.log_feedback(player.get_current_song(), feedback)
                except KeyboardInterrupt:
                    raise  # Allow Ctrl+C
                except:
                    pass  # Skip if no input

            song = player.get_current_song()  # Get current song for logging
            logger.log_entry(state, features, song)  # Log data for analysis (Week 2: Scientific coding)
            print(f"Current state: {state}, Playing: {song}")  # Console feedback
            
            # Periodic feedback (every 24 cycles, ~2 minutes)
            feedback_counter += 1
            if feedback_counter >= 24:
                feedback_counter = 0
                try:
                    feedback = input("Overall, like the current song? (y/n): ").strip().lower()
                    logger.log_feedback(song, feedback)
                except KeyboardInterrupt:
                    raise
                except:
                    pass  # Skip if no input

            time.sleep(5)  # Wait for next 5-second window (real-time rolling analysis)
    except KeyboardInterrupt:
        print("Session ended by user.")# Run the session when script is executed
if __name__ == "__main__":
    run_session()