# Logger module for recording session data
# Logs EEG features, states, and music for future ML analysis (Week 2: Scientific coding)

import csv  # For CSV writing
import time  # For timestamps
from app_configuration import LOG_FILE  # Import log file path

class Logger:
    # Initializes the logger and writes CSV header
    def __init__(self):
        self.log_file = LOG_FILE  # Path to CSV file
        try:
            with open(self.log_file, 'x', newline='') as f:  # Try to create new file
                writer = csv.writer(f)
                writer.writerow(["timestamp", "state", "alpha_power", "beta_power", "theta_power", "beta_alpha", "alpha_theta", "alpha_beta", "theta_alpha", "song"])  # Header row
        except FileExistsError:
            pass  # File exists, don't overwrite header

    # Logs a single entry with timestamp and data
    def log_entry(self, state, features, song):
        try:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")  # Current time (Week 2: Reproducible logging)
            with open(self.log_file, 'a', newline='') as f:  # Append mode
                writer = csv.writer(f)
                writer.writerow([  # Write row with all data
                    timestamp, state,
                    features["alpha_power"], features["beta_power"], features["theta_power"],
                    features["beta_alpha"], features["alpha_theta"], features["alpha_beta"], features["theta_alpha"], song
                ])
        except Exception as e:
            print(f"Logging error: {e}")

    # Logs user feedback for song evaluation
    def log_feedback(self, song, feedback):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        feedback_file = "feedback_log.csv"
        # Create header if file doesn't exist
        try:
            with open(feedback_file, 'x', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "song", "feedback"])
        except FileExistsError:
            pass
        with open(feedback_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, song, feedback])