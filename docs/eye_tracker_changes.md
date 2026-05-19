# Eye Tracker — Technical Documentation

## Files Changed
- `src/vision/eye_tracker.py`
- `src/vision/gaze_state.py`

---

## 1. Preprocessing Pipeline
**Why:** Raw webcam frames cause mediapipe to miss eyes behind glasses 
glare or in low contrast conditions.

**What was added:** `_preprocess(frame)` runs on every frame before 
feature extraction:
- Bilateral filter removes grain without blurring the iris edge
- CLAHE enhances contrast on the brightness channel only (not colour)

**Inspiration:** GazeTracking (github.com/antoinelame/GazeTracking)

---

## 2. Blink Rate (BPM)
**Why:** Blink rate is a clinical signal relevant to attention research. 
The original code detected blinks but discarded that data.

**What was added:**
- Rising edge detection counts each blink once at onset
- Rolling 60-second window gives live BPM
- Displayed in HUD alongside blink state

---

## 3. Fixation Detection
**Why:** Fixation duration is one of the most studied signals in ADHD 
research. Short fixations and frequent saccades are known indicators.

**What was added:**
- `is_fixating` field in `GazeState`
- Set to `True` when gaze moves less than 30px from last frame
- Cursor ring changes colour: green = fixating, amber = moving

---

## 4. Cursor Alpha Fix
**Why:** Original code computed `cursor_alpha` but never applied it — 
cursor was always fully opaque regardless of tracking confidence.

**What was fixed:** Cursor now uses `SRCALPHA` surface so alpha 
actually renders. Cursor fades out when face is lost.

---

## 5. 9-Point Calibration
**Why:** 5-point calibration only covers corners and centre. 
9-point covers more of the screen giving better spatial accuracy.

**What was changed:** `create_model()` now calls 
`run_9_point_calibration()` instead of `run_5_point_calibration()`.

---

## 6. `reset()` Method
**Why:** Without reset, running two sessions back to back would 
carry over BPM counts and smoother state from the previous session.

**What was added:** `reset()` clears all per-trial state while 
keeping the trained model loaded.

---

## Test Coverage
Run: `python tests/test_eye_tracker.py`

| Test | What it checks |
|---|---|
| `test_gaze_state_defaults` | GazeState initialises safely |
| `test_is_fixating_field_exists` | New field works correctly |
| `test_cursor_alpha_bounds` | Alpha never goes out of 0–1 range |
| `test_blink_rising_edge` | One blink = one count, not one per frame |
| `test_fixation_detection` | 30px threshold works correctly |