import cv2
import pygame
import numpy as np
import time
import collections

from src.config import *
from eyetrax.calibration import (
    run_9_point_calibration,
)
from eyetrax.filters import (
    KalmanSmoother,
    make_kalman,
)
from eyetrax.gaze import GazeEstimator
from src.vision.gaze_state import GazeState

# Created once at module level — expensive to initialise per frame
_CLAHE = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))


def _preprocess(frame):
    """
    Bilateral filter + CLAHE on luminance channel.
    Improves iris detection for glasses wearers and small/squinted eyes.
    - Bilateral filter removes noise without blurring the iris edge
    - CLAHE only on L channel (brightness) — correct, not on colour
    Inspired by GazeTracking (github.com/antoinelame/GazeTracking)
    """
    frame = cv2.bilateralFilter(frame, d=7, sigmaColor=50, sigmaSpace=50)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    lab = cv2.merge([_CLAHE.apply(l), a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


class EyeTracker:
    def __init__(self):
        self.gaze_estimator = GazeEstimator()
        self.current_state = GazeState()

        # rolling blink log → BPM over last 60 s
        self._blink_log = collections.deque()
        self._prev_blink = False

        self._set_smoother()

    # -----------------------------
    # MODEL
    # -----------------------------

    def create_model(self, path):
        try:
            # 9-point gives better spatial coverage than 5-point
            run_9_point_calibration(self.gaze_estimator)
            self.gaze_estimator.save_model(path)
            self.gaze_estimator.load_model(path)
            print(f"[EyeTracker] Model created: {path}")
        except Exception as e:
            print(f"[EyeTracker] \033[91mError:\033[0m create_model failed: {e}")
            raise  # surface the failure — caller must know it failed

    def load_model(self, path):
        try:
            self.gaze_estimator.load_model(path)
            print(f"[EyeTracker] Loaded model: {path}")
        except Exception as e:
            print(f"[EyeTracker] \033[91mError:\033[0m load_model failed: {e}")
            raise

    # -----------------------------
    # FILTER
    # -----------------------------

    def _set_smoother(self):
        self.kalman = make_kalman()
        self.smoother = KalmanSmoother(self.kalman)

    # -----------------------------
    # RESET
    # -----------------------------

    def reset(self):
        """Clear per-trial state without losing the loaded model.
        Call this between sessions or game rounds."""
        self.current_state = GazeState()
        self._blink_log.clear()
        self._prev_blink = False
        self._set_smoother()

    # -----------------------------
    # UPDATE
    # -----------------------------

    def update(self, frame) -> GazeState:
        if frame is None:
            return self.current_state

        # preprocess before feature extraction (glasses/small eyes support)
        processed = _preprocess(frame)

        s = self.current_state

        try:
            features, blink_detected = self.gaze_estimator.extract_features(processed)
        except Exception:
            return s

        s.blink_detected = blink_detected

        # track blink onsets (rising edge only) for BPM
        # rising edge = only the moment blinking starts, not every blink frame
        if blink_detected and not self._prev_blink:
            self._blink_log.append(time.time())
        self._prev_blink = blink_detected

        # drop blinks older than 60 seconds from the rolling window
        now = time.time()
        while self._blink_log and self._blink_log[0] < now - 60:
            self._blink_log.popleft()

        if features is not None and not blink_detected:
            try:
                gaze_point = self.gaze_estimator.predict(np.array([features]))[0]
                x, y = map(int, gaze_point)

                # store previous position before updating
                prev_x = s.pred_x if s.pred_x is not None else x
                prev_y = s.pred_y if s.pred_y is not None else y

                s.pred_x, s.pred_y = self.smoother.step(x, y)
                s.cursor_alpha = min(s.cursor_alpha + CURSOR_STEP, 1.0)

                # fixation: gaze hasn't moved much from last position
                s.is_fixating = (
                    abs(x - prev_x) < 30 and
                    abs(y - prev_y) < 30
                )

            except Exception:
                s.pred_x = s.pred_y = None
                s.cursor_alpha = max(s.cursor_alpha - CURSOR_STEP, 0.0)
                s.is_fixating = False
        else:
            s.pred_x = s.pred_y = None
            s.cursor_alpha = max(s.cursor_alpha - CURSOR_STEP, 0.0)
            s.is_fixating = False

        return s

    # -----------------------------
    # RENDER
    # -----------------------------

    def render(self, screen, frame, current_state, font):
        if frame is None:
            return

        # flip for display only — update() already saw the preprocessed frame
        frame = cv2.flip(frame, 1)

        # gaze cursor with alpha fade + fixation ring
        if (
            current_state.pred_x is not None
            and current_state.pred_y is not None
            and current_state.cursor_alpha > 0
        ):
            cx = int(current_state.pred_x)
            cy = int(current_state.pred_y)
            radius = 15
            alpha_int = int(current_state.cursor_alpha * 255)

            # SRCALPHA surface so the alpha fade actually works
            surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(
                surf, (*COLORS[4][:3], alpha_int), (radius, radius), radius
            )

            # outer ring: green = fixating, amber = moving
            if current_state.is_fixating:
                ring_col = (60, 220, 120, alpha_int // 2)
            else:
                ring_col = (220, 160, 40, alpha_int // 3)

            pygame.draw.circle(surf, ring_col, (radius, radius), radius, 3)
            screen.blit(surf, (cx - radius, cy - radius))

        # camera thumbnail
        try:
            thumb = cv2.resize(frame, (CAM_WIDTH, CAM_HEIGHT))
            thumb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB).swapaxes(0, 1)
            screen.blit(
                pygame.surfarray.make_surface(thumb),
                (screen.get_width() - CAM_WIDTH - MARGIN,
                 screen.get_height() - CAM_HEIGHT - MARGIN),
            )
        except Exception:
            pass

        # HUD: blink state + rolling BPM
        bpm = len(self._blink_log)
        if current_state.blink_detected:
            blink_txt = f"Blinking   BPM: {bpm}"
            blink_clr = (80, 80, 240)
        else:
            blink_txt = f"Eyes open  BPM: {bpm}"
            blink_clr = (60, 210, 100)

        screen.blit(font.render(blink_txt, True, blink_clr), (50, 100))
