# Neuro-Adaptive Music Player

A Python-based desktop application that uses real-time EEG analysis to detect mental states (focus, relax, fatigue, happy, sad) and adapts music playback accordingly using AI recommendations. Built with signal processing techniques from CMP9780M lectures.

## Overview
This app demonstrates applied signals processing by:
- Processing EEG data in 5-second windows.
- Classifying brain states via band power ratios.
- Recommending and playing music tailored to each state.
- Logging data for analysis and learning user preferences.

## Features
- **EEG Simulation/Streaming**: Simulated data for demo; commented code for real devices (Muse, OpenBCI, NeuroSky).
- **State Detection**: Identifies 5 states using thresholds on alpha/beta/theta powers (Week 2-4).
- **Enhanced Emotion Recognition**: Optional deep learning mode with CNN+BiLSTM architecture and Frontal Alpha Asymmetry (FAA) for 70-90% accuracy.
- **AI Music Recommendations**: OpenAI suggests 3 songs per state; plays first, allows skipping to alternatives (agentic AI for user preference).
- **Playback Options**: Local MP3 files (pygame) or Amazon Music search (browser).
- **Feedback System**: Immediate prompts for likes/skips; logs for refining recommendations.
- **Logging**: CSV files for sessions and feedback.
- **Headless Operation**: Runs in console for continuous adaptation.

## Requirements
- Python 3.x
- Libraries: numpy, scipy, pygame, openai, requests
- Install: `pip install -r requirements.txt`
- **Optional (Deep Learning)**: TensorFlow, scikit-learn for CNN+BiLSTM emotion recognition
  - Install: `pip install tensorflow scikit-learn`
- OpenAI API key (free tier available)

## Setup
1. Clone/download the folder.
2. Install dependencies.
3. Add OpenAI API key to `app_configuration.py`.
4. For local music: Add MP3s to `music/` (name as "Song by Artist.mp3").
5. For Amazon: Set `USE_LOCAL = False`.
6. Run `python neuro_adaptive_app.py`.

## How It Works
### EEG Processing
- Simulates/streams EEG data (256 Hz).
- Applies 1-30 Hz bandpass filter (Week 3).
- Computes PSD (Welch, Week 2) for band powers.
- **Standard Mode**: Calculates ratios (e.g., beta/alpha >1.5 = focus).
- **Deep Learning Mode**: Uses CNN+BiLSTM with Frontal Alpha Asymmetry for improved accuracy (70-90% vs 60-75%).
- Classifies state.

### Deep Learning Enhancement (Optional)
- **Architecture**: CNN extracts spatial-frequency features, BiLSTM captures temporal context.
- **Frontal Alpha Asymmetry (FAA)**: Measures left/right frontal activation for emotional valence (based on Frantzidis et al., 2010).
- **Multi-band features**: Delta, Theta, Alpha, Beta, Gamma powers per channel.
- **Benefits**: Higher accuracy, reduced confusion between similar states (happy/sad).
- **Details**: See `dl_enhancement_docs.md` for full technical documentation.

### State-to-Music Mapping
- States: Focus (high beta), Relax (high alpha), Fatigue (low power), Happy (high alpha/beta), Sad (high theta/alpha).
- AI recommends 3 songs via prompt (variety but state-aligned).
- Plays first song; user can skip to next or re-recommend if all disliked.

### Playback & Adaptation
- Local: Searches `music/`, plays with fade.
- Amazon: Opens search URL.
- Adapts every 5s if state changes; immediate feedback on new songs.

### Feedback & Learning
- Prompts for y/n every ~15s.
- Logs to `feedback_log.csv`.
- Analyze logs to see patterns (e.g., 80% likes for "Happy" songs in happy state).

## Configuration
Edit `app_configuration.py`:
- Thresholds for states.
- USE_LOCAL for playback mode.
- API key and model.

## Files
- `neuro_adaptive_app.py`: Main app entry point and session loop.
- `eeg_data_simulator.py`: Simulates or streams EEG data.
- `eeg_signal_processor.py`: Processes signals and detects mental states (enhanced with optional deep learning).
- `dl_emotion_model.py`: Deep learning module with CNN+BiLSTM and FAA (optional).
- `ai_music_controller.py`: AI-driven music recommendations and playback.
- `session_data_logger.py`: Logs session data and feedback.
- `app_configuration.py`: Configuration settings and thresholds.
- `requirements.txt`: Python dependencies.
- `dl_enhancement_docs.md`: Technical documentation for deep learning features.
- `session_log.csv`: Generated session data.
- `feedback_log.csv`: Generated user feedback.

## Usage Examples
- Run headless: `python neuro_adaptive_app.py` (Ctrl+C to stop).
- Monitor console for states/songs.
- Review logs for insights.

## Recent Improvements
- **Error Handling**: Added try-except blocks for API calls, file operations, and pygame initialization to prevent crashes.
- **Robust Parsing**: Improved OpenAI response parsing for song recommendations to handle varied formats.
- **Logging Safety**: Logger now appends to existing files without overwriting headers.
- **Feedback Timing**: Switched to cycle-based periodic feedback for consistency.
- **Signal Processing Checks**: Added validation for PSD computation to avoid errors with short signals.
- **Graceful Shutdown**: Better handling of Ctrl+C interrupts in the main loop.
- **Deep Learning Integration**: Added optional CNN+BiLSTM model with Frontal Alpha Asymmetry for 70-90% accuracy (based on Frantzidis et al., 2010).
- **Backward Compatibility**: Graceful fallback to threshold-based classification if deep learning unavailable.

## EEG Sticker Add-on for Any Headset — Quick Integration Plan
### Overview
Upgrade any commercial (Bluetooth or wired) headset with EEG capability using easy-apply dry sticker electrodes and a USB-C-connected mini EEG board. No destructive modifications—universal, comfortable, and affordable.

### Parts Needed
- 2–4 dry sticker EEG electrodes (with skin-safe adhesive)
- Flexible headband cable or individual electrode leads
- Open-source EEG amplifier module (e.g., ADS1299, ADS1115)
- USB-C microcontroller (e.g., Teensy 4.0, ESP32-S2/S3)
- USB-C cable (for power and data)
- Small enclosure for electronics (can be 3D-printed or repurposed)
- Wires, adhesive pads, and basic tools

### How to Assemble
**Apply Electrodes:**
- Peel and stick 2–4 electrodes to the underside of the headband where it touches your scalp/forehead.

**Wire Up:**
- Route electrode wires under the headband padding to one side or ear cup.

**Connect EEG Module:**
- Attach wires to the amplifier+microcontroller module in the enclosure.
- Place the enclosure on or inside the ear cup, or clip on the headband.

**Plug Into PC:**
- Use a USB-C cable to connect your headset to your computer (for both headset charging and EEG data transfer).

**Run EEG Software:**
- Your neuro-adaptive music/focus app receives live EEG over USB.
- Use/replace sticker electrodes as desired (weeks of use per pack).

### How it Works
- Sticker electrodes pick up brain signals (focus, fatigue, relaxation) through your scalp.
- The EEG board digitizes and streams these signals over USB.
- Your software detects cognitive state and adapts music/playlists, logs data, or provides feedback in real time.

### Estimated Cost
- Total kit cost (prototype/small batch): £30–£55
  - Electrodes ~£3, microcontroller/amp ~£22–£40, case/cable/wires ~£5–£12
- Sticker electrodes: ~£0.40–£1 each (replaced every few weeks/months)
- No need to modify your headphones: Plug-and-play with any set.

### Key Features
- **Universal**: Works with nearly all consumer headsets (gaming/music)
- **Comfortable & Unintrusive**: No bulky hardware, no skin preparation, nothing visible when worn
- **USB-C Powered/Data**: No batteries needed, one cable for everything
- **Easy Maintenance**: Swap stickers as needed, wipe headband as normal
- Upgrade your headset to a brain-driven device—perfect for DIY neurotech, research, productivity tools, and neuro-adaptive music or game integration!

## License
Proprietary license. See root LICENSE file. Commercial use, especially for neural/EEG applications, requires explicit permission. Educational use only for CMP9780M assessment.