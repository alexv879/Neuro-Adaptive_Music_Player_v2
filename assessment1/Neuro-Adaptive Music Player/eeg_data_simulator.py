# EEG simulator module for generating demo EEG data
# Simulates clinical EEG signals for testing without real hardware (Week 1: Signal fidelity and simulation)
# Uncomment and modify the REAL EEG sections below to use actual devices (e.g., Muse, OpenBCI)

import numpy as np  # For numerical arrays and sine waves
import time  # For simulating state changes over time

from app_configuration import FS, WINDOW_SIZE  # Import sampling rate and window size

class EEGSimulator:
    # Initializes the simulator with sampling parameters
    def __init__(self):
        self.fs = FS  # Sampling frequency (256 Hz, Week 1: Nyquist-compliant)
        self.window_samples = WINDOW_SIZE * self.fs  # Samples per 5-second window
        self.current_state = "relax"  # Initial simulated state
        
        # REAL EEG: Uncomment for device integration
        # self.device = None  # Placeholder for device object
        # self.connect_device()  # Call to connect (uncomment below)

    # Generates a 5-second window of simulated EEG data
    def get_data_stream(self):
        # SIMULATED EEG: Use this for demo
        # Create time array for the window
        t = np.arange(self.window_samples) / self.fs  # Time vector (Week 1: Discrete time sampling)
        
        # Simulate EEG bands: Alpha (8-12 Hz), Beta (12-30 Hz), Theta (4-8 Hz)
        alpha = np.sin(2 * np.pi * 10 * t) * 10  # Alpha wave (µV amplitude, Week 2: EEG frequency bands)
        beta = np.sin(2 * np.pi * 20 * t) * 5   # Beta wave
        theta = np.sin(2 * np.pi * 6 * t) * 8   # Theta wave
        noise = np.random.normal(0, 2, self.window_samples)  # Add Gaussian noise (Week 3: Noise in signals)
        
        # Combine into raw signal
        signal = alpha + beta + theta + noise
        
        # Simulate dynamic state changes based on time (for demo)
        if time.time() % 30 < 10:  # First 10s of 30s cycle: Focus
            self.current_state = "focus"
            signal += beta * 2  # Boost beta for focus state (Week 2: Band power changes)
        elif time.time() % 30 < 20:  # Next 10s: Relax
            self.current_state = "relax"
        else:  # Last 10s: Fatigue
            self.current_state = "fatigue"
            signal *= 0.5  # Reduce amplitude for fatigue (Week 4: Artifact-like changes)
        
        return signal, self.current_state  # Return data and simulated state
        
        # REAL EEG: Uncomment and modify for actual device streaming
        # Example for Muse SDK (install muse-python first: pip install muselsl)
        # from muselsl import stream, list_muses
        # if self.device is None:
        #     muses = list_muses()
        #     if muses:
        #         stream(muses[0]['address'], ppg=False, acc=False, gyro=False)  # Start streaming EEG only
        #         self.device = muses[0]  # Store device info
        # # Collect data from stream (requires additional setup for real-time buffering)
        # # signal = get_eeg_data_from_muse(self.device, self.window_samples)  # Custom function to fetch
        # # self.current_state = "unknown"  # Or detect from data
        # # return signal, self.current_state
        
        # Example for OpenBCI (install pyopenbci: pip install pyopenbci)
        # from pyopenbci import OpenBCICyton
        # if self.device is None:
        #     self.device = OpenBCICyton(port='COM3')  # Adjust port
        #     self.device.start_stream(self.handle_sample)  # Start streaming
        # # In handle_sample callback: collect samples into buffer
        # # signal = np.array(self.buffer[-self.window_samples:])  # Last window
        # # self.current_state = "unknown"
        # # return signal, self.current_state
        
        # Example for NeuroSky MindWave (install mindwave-python: pip install mindwave)
        # import mindwave
        # if self.device is None:
        #     self.device = mindwave.Headset('/dev/tty.MindWave')  # Adjust port
        #     self.device.connect()
        # # Read data in loop
        # # signal = []  # Collect samples
        # # for _ in range(self.window_samples):
        # #     signal.append(self.device.raw_value)  # Raw EEG
        # # signal = np.array(signal)
        # # self.current_state = "unknown"
        # # return signal, self.current_state

    # REAL EEG: Helper method to connect device (uncomment and adapt)
    # def connect_device(self):
    #     # Add connection logic here, e.g., for Muse:
    #     # from muselsl import list_muses
    #     # muses = list_muses()
    #     # if muses:
    #     #     print("Muse found:", muses[0])
    #     #     self.device = muses[0]
    #     # else:
    #     #     print("No Muse detected")
    #     pass