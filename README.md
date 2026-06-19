# ADHD Attention Tracker

A webcam-based eye-tracking game that measures attention signals — fixation stability, blink rate, and gaze patterns — while the player works through a Simon-Says-style memory challenge. Built with Python, OpenCV/MediaPipe, and Pygame, with optional physical hardware support via an Arduino + Wokwi simulation bridge.

The idea: instead of a clinical questionnaire, give the player a simple memory game, track where their eyes actually go while they play, and surface attention-relevant metrics (fixation ratio, blink rate, gaze stability) in a session summary at the end.

## How it works

1. **Login** — players sign in with a username/password. New names get a profile created automatically (`src/core/profile_manager.py`), each with its own data folder for sessions, heatmaps, and a personal gaze model.
2. **Calibration** — first-time users run a 9-point calibration (via [`eyetrax`](https://pypi.org/project/eyetrax/)) so the tracker can map raw gaze features to screen coordinates. The trained model is saved per-user and reloaded automatically on future sessions.
3. **Play** — a 4-color memory game (think Simon Says): the game flashes a growing sequence, the player repeats it back by clicking on-screen buttons *or* pressing physical buttons wired to an Arduino. Wrong move ends the round; correct sequences get longer.
4. **Track** — while playing, a background thread reads webcam frames, extracts gaze features, and classifies fixations (Pupil Labs–style dispersion/duration model) vs. blinks vs. saccades, all rendered live as a gaze cursor overlay.
5. **Review** — on quit, the session is saved and a heatmap of gaze positions is generated over a screenshot of the game screen, alongside a summary screen showing session duration, fixation ratio, total fixation time, blink rate (BPM), and a few rule-based suggestions.

## Features

- **Real-time gaze tracking** — webcam-based gaze estimation via `eyetrax` + MediaPipe, with a preprocessing pipeline (bilateral filtering + CLAHE) to improve iris detection for glasses wearers and low-light conditions.
- **Fixation & blink detection** — dispersion/duration-based fixation classification and rising-edge blink counting, producing a live blink-rate (BPM) readout — both signals commonly studied in attention research.
- **Per-user profiles & gaze models** — each player gets their own calibration model, session logs, and heatmaps, stored under `data/<username>_<id>/`.
- **Session recording & heatmaps** — every session's gaze trail is logged and rendered as a heatmap over the actual gameplay screen.
- **Hybrid input** — play with mouse clicks on-screen or with real push-buttons on an Arduino, bridged over serial (`pyserial`) so the same game logic handles both.
- **Hardware-free testing** — a bundled [Wokwi](https://wokwi.com/) simulation server (`wokwi_simulate_server/`) lets you simulate the Arduino circuit and test the full input pipeline with no physical hardware attached.
- **Configurable calibration & smoothing** — supports multiple calibration patterns (5-point, 9-point, dense grid, Lissajous) and several smoothing filters (Kalman, Kalman+EMA, KDE, none) for the gaze signal.

## Project structure

```
adhd-project/
├── main.py                      # Entry point — launches GameManager
├── requirements.txt
├── src/
│   ├── config.py                # Screen size, colors, paths, tracker constants
│   ├── core/
│   │   ├── game_manager.py      # Top-level state machine (LOGIN → MENU → PLAYING)
│   │   ├── engine.py            # Memory-game logic + Arduino serial bridge
│   │   ├── profile_manager.py   # User accounts, per-user data folders
│   │   ├── session_recorder.py  # Gaze logging + heatmap generation
│   │   └── user.py / game_state.py
│   ├── vision/
│   │   ├── eye_tracker.py       # Gaze estimation, fixation/blink detection
│   │   ├── camera_threading.py  # Threaded webcam capture
│   │   └── gaze_state.py        # Per-frame gaze state dataclass
│   ├── ui/                      # Pygame-GUI scenes (login, menu, HUD)
│   ├── game_logic/               # Game sprites (color buttons)
│   ├── hardware/                # Serial controller for the Arduino buttons
│   └── common/                  # Event manager, JSON helpers
├── arduino/                     # PlatformIO project for the physical button box
│   └── src/main.cpp             # Reads 4 buttons, sends color names over serial
├── wokwi_simulate_server/       # Wokwi-CLI gateway for simulating the Arduino
├── docs/                        # Technical notes (e.g. eye-tracker design decisions)
├── tests/                       # Unit tests for the eye tracker
└── data/                        # Per-user profiles, models, sessions, heatmaps (generated)
```

## Getting started

### Requirements
- Python 3.10+
- A webcam
- (Optional) An Arduino with 4 push-buttons, or just use the bundled Wokwi simulator

### Installation

```bash
git clone https://github.com/tushi468/adhd-project.git
cd adhd-project
pip install -r requirements.txt
```

### Run

```bash
python main.py
```

On first login you'll be guided through a 9-point calibration. After that, just log in and play — your gaze model is reloaded automatically.

### Using the physical controller (optional)

The `arduino/` folder is a [PlatformIO](https://platformio.org/) project that reads 4 push-buttons and writes `RED` / `GREEN` / `BLUE` / `YELLOW` over serial at 115200 baud. To test it without hardware, the bundled Wokwi gateway (`wokwi_simulate_server/`) simulates the circuit defined in `arduino/diagram.json` and exposes it over `rfc2217://localhost:4000`, which `src/core/engine.py` connects to automatically if available — the game falls back to mouse input if no serial connection is found.

## Tests

```bash
python tests/test_eye_tracker.py
```

Covers gaze-state defaults, fixation detection thresholds, blink rising-edge counting, and cursor alpha bounds. See `docs/eye_tracker_changes.md` for the reasoning behind the tracker's design.

## Tech stack

| Layer | Tools |
|---|---|
| Game / rendering | `pygame-ce`, `pygame-gui` |
| Vision / gaze estimation | `opencv-python`, `mediapipe`, `eyetrax` |
| Data | `numpy`, `pandas` |
| Hardware bridge | `pyserial`, PlatformIO, Wokwi |

## Disclaimer

This is an attention-tracking demo built for a class/personal project, **not a diagnostic or medical tool**. Fixation ratio, blink rate, and other metrics shown in the summary screen are informational signals drawn from attention-research literature, not a clinical assessment of ADHD or any other condition.

## License

MIT — see [LICENSE](./LICENSE). (The `wokwi_simulate_server/` component carries its own MIT license from its original author.)
